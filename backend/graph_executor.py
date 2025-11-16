"""
Graph Executor - True graph-based execution engine for node workflows.

Architecture:
- GraphExecutor: orchestrates execution by following actual graph connections
- Nodes are executed only when their inputs are ready
- Data flows through connections naturally
- Only data reaching Output node is saved
"""

import logging
import time
import re
from typing import Dict, List, Optional, Set, Tuple
from database import ExecutionManager, AsyncSessionManager
from utils.image_utils import load_image

logger = logging.getLogger(__name__)


# ============================================================================
# Graph Topology Helper
# ============================================================================

class GraphTopology:
    """Analyzes graph structure and provides connection information."""
    
    def __init__(self, nodes: List[Dict], connections: List[Dict]):
        self.nodes = {n['id']: n for n in nodes}
        self.connections = connections
        
        # Build adjacency map for fast lookups
        self.outgoing = {}  # node_id -> [(target_id, from_port, to_port)]
        self.incoming = {}  # node_id -> [(source_id, from_port, to_port)]
        
        for conn in connections:
            from_id = conn['from']
            to_id = conn['to']
            from_port = conn.get('fromPort', 0)
            to_port = conn.get('toPort', 0)
            
            if from_id not in self.outgoing:
                self.outgoing[from_id] = []
            self.outgoing[from_id].append((to_id, from_port, to_port))
            
            if to_id not in self.incoming:
                self.incoming[to_id] = []
            self.incoming[to_id].append((from_id, from_port, to_port))
    
    def get_downstream_nodes(self, node_id: str, from_port: int = None) -> List[Tuple[str, int, int]]:
        """Get all nodes connected to this node's outputs."""
        edges = self.outgoing.get(node_id, [])
        if from_port is not None:
            edges = [(tid, fp, tp) for tid, fp, tp in edges if fp == from_port]
        return edges
    
    def get_upstream_nodes(self, node_id: str, to_port: int = None) -> List[Tuple[str, int, int]]:
        """Get all nodes connected to this node's inputs."""
        edges = self.incoming.get(node_id, [])
        if to_port is not None:
            edges = [(sid, fp, tp) for sid, fp, tp in edges if tp == to_port]
        return edges
    
    def find_node_by_type(self, node_type: str) -> Optional[Dict]:
        """Find first node of given type."""
        return next((n for n in self.nodes.values() if n['type'] == node_type), None)


class PromptResolver:
    """
    Handles all prompt and conjunction resolution logic.
    
    Optimizations:
    - Indexed connection lookups (O(1) instead of O(n))
    - Cached node lookups by ID and type
    - Single-pass connection analysis
    - Unified input source resolution
    """
    
    def __init__(self, nodes: List[Dict], connections: List[Dict]):
        # Build indexed lookups for fast access
        self.nodes_by_id = {n['id']: n for n in nodes}
        self.nodes_by_type = {}
        for node in nodes:
            node_type = node['type']
            if node_type not in self.nodes_by_type:
                self.nodes_by_type[node_type] = []
            self.nodes_by_type[node_type].append(node)
        
        # Index connections by target node for O(1) lookup
        self.inputs_by_node = {}  # node_id -> [(from_node, from_port, to_port)]
        for conn in connections:
            target_id = conn['to']
            if target_id not in self.inputs_by_node:
                self.inputs_by_node[target_id] = []
            
            from_node = self.nodes_by_id.get(conn['from'])
            if from_node:
                self.inputs_by_node[target_id].append({
                    'node': from_node,
                    'from_port': conn.get('fromPort', 0),
                    'to_port': conn.get('toPort', 0)
                })
        
        # Cache for static prompts
        self._prompt_cache = {}
    
    def get_static_prompt(self, ai_node: Dict) -> str:
        """Get static prompt connected to AI node (cached)."""
        node_id = ai_node['id']
        if node_id in self._prompt_cache:
            return self._prompt_cache[node_id]
        
        # Check inputs for prompt node
        inputs = self.inputs_by_node.get(node_id, [])
        for input_info in inputs:
            if input_info['node']['type'] == 'prompt':
                prompt = input_info['node']['data'].get('text', '')
                self._prompt_cache[node_id] = prompt
                return prompt
        
        self._prompt_cache[node_id] = ''
        return ''
    
    def get_prompt_for_image(self, ai_node: Dict, prev_captions: List[str], 
                            img_index: int, base_prompt: str) -> Optional[str]:
        """
        Get final prompt for specific image (handles chaining/conjunctions).
        
        Priority order:
        1. Previous AI caption (port 1 connection from aimodel)
        2. Conjunction node output
        3. Static base prompt
        """
        node_id = ai_node['id']
        inputs = self.inputs_by_node.get(node_id, [])
        
        # Check for caption chaining or conjunction on prompt port (port 1)
        for input_info in inputs:
            source_node = input_info['node']
            source_type = source_node['type']
            to_port = input_info['to_port']
            
            # Caption chaining: previous AI → prompt port
            if to_port == 1 and source_type == 'aimodel':
                caption = prev_captions[img_index]
                return caption if caption else None
            
            # Conjunction feeding prompt
            if source_type == 'conjunction':
                return self._resolve_conjunction(source_node, prev_captions, img_index)
        
        # Fall back to static prompt
        return base_prompt or None
    
    def resolve_output(self, last_ai: Dict, output_node: Dict, 
                      prev_captions: List[str], img_index: int) -> str:
        """
        Resolve final output (handles output conjunction if present).
        
        Checks if there's a conjunction node between last AI and output.
        """
        output_id = output_node['id']
        inputs = self.inputs_by_node.get(output_id, [])
        
        # Check for conjunction feeding output
        for input_info in inputs:
            source_node = input_info['node']
            if source_node['type'] == 'conjunction':
                # Verify this conjunction is fed by the last AI
                conj_inputs = self.inputs_by_node.get(source_node['id'], [])
                if any(inp['node']['id'] == last_ai['id'] for inp in conj_inputs):
                    return self._resolve_conjunction(source_node, prev_captions, img_index)
        
        # No conjunction, return caption directly
        return prev_captions[img_index]
    
    def _resolve_conjunction(self, conj_node: Dict, prev_captions: List[str], 
                            img_index: int) -> str:
        """
        Resolve conjunction template with actual values.
        
        Replaces {key} placeholders in template with connected items or captions.
        """
        template = conj_node['data'].get('template', '')
        if not template:
            return prev_captions[img_index]
        
        # Build reference map from connected items
        connected_items = conj_node['data'].get('connectedItems', [])
        ref_map = {
            item.get('refKey', ''): (
                prev_captions[img_index] 
                if item.get('content', '') == '[Generated Captions]' 
                else item.get('content', '')
            )
            for item in connected_items
        }
        
        # Substitute {key} placeholders in one pass
        resolved = re.sub(
            r'\{([^}]+)\}', 
            lambda m: ref_map.get(m.group(1), m.group(0)), 
            template
        )
        
        return resolved


class ExecutionState:
    """
    Simple execution state tracker - inspired by LangGraph.
    
    Tracks basic counters without complex stage logic.
    Updates happen naturally as nodes execute.
    """
    
    def __init__(self, exec_manager: ExecutionManager, job_id: str, total_images: int):
        self.exec_manager = exec_manager
        self.job_id = job_id
        self.total_images = total_images
        self.start_time = time.time()
        self.processed = 0
        self.success = 0
        self.failed = 0
    
    def increment(self, success: bool = True):
        """Increment counters when an image is processed."""
        self.processed += 1
        if success:
            self.success += 1
        else:
            self.failed += 1
    
    def update(self):
        """Send current state to database."""
        elapsed = time.time() - self.start_time
        speed = self.processed / elapsed if elapsed > 0 else 0
        
        self.exec_manager.update_status(
            self.job_id,
            processed=self.processed,
            success=self.success,
            failed=self.failed,
            progress={
                'speed': f"{speed:.1f} img/s" if speed > 0 else "",
                'progress': f"{self.processed}/{self.total_images}"
            }
        )


# ============================================================================
# Main Executor - Graph Execution
# ============================================================================

class GraphExecutor:
    """
    True graph executor - follows actual connections, not predefined chains.
    
    Execution flow:
    1. Start at Input node with images
    2. Follow connections to see what's downstream
    3. Execute nodes when their data is ready
    4. Pass data through connections
    5. Save only what reaches Output node
    """
    
    def __init__(self, exec_manager: ExecutionManager, async_session: AsyncSessionManager):
        self.exec_manager = exec_manager
        self.async_session = async_session
        self.should_cancel = False
    
    async def execute(self, job_id: str, get_model_func) -> None:
        """Execute a job by following the graph structure."""
        job = self.exec_manager.get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return
        
        try:
            self.exec_manager.update_status(job_id, status='running')
            
            graph = job['graph']
            image_ids = job['image_ids']
            nodes = graph.get('nodes', [])
            connections = graph.get('connections', [])
            
            # Build topology
            topology = GraphTopology(nodes, connections)
            
            # Find Input and Output nodes
            input_node = topology.find_node_by_type('input')
            output_node = topology.find_node_by_type('output')
            
            if not input_node or not output_node:
                raise ValueError("Graph must have Input and Output nodes")
            
            # Check if there's any path from Input to Output
            if not self._has_path_to_output(input_node['id'], output_node['id'], topology):
                raise ValueError("No path from Input to Output node")
            
            # Execute the graph
            await self._execute_graph(
                job_id, image_ids, topology, input_node, output_node, get_model_func
            )
            
            self.exec_manager.update_status(job_id, status='completed')
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.exception(f"Job {job_id} failed: {e}")
            self.exec_manager.update_status(job_id, status='failed', error=str(e))
    
    def _has_path_to_output(self, start_id: str, output_id: str, topology: GraphTopology) -> bool:
        """Check if there's any path from start to output."""
        visited = set()
        queue = [start_id]
        
        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            
            if current_id == output_id:
                return True
            
            for target_id, _, _ in topology.get_downstream_nodes(current_id):
                if target_id not in visited:
                    queue.append(target_id)
        
        return False
    
    def cancel(self):
        """Signal cancellation of current execution."""
        self.should_cancel = True
    
    
    async def _execute_graph(
        self, job_id: str, image_ids: List[str], topology: GraphTopology,
        input_node: Dict, output_node: Dict, get_model_func
    ) -> None:
        """
        Execute graph by following connections from Input to Output.
        
        This is the TRUE graph execution - we traverse the actual graph structure.
        """
        prompt_resolver = PromptResolver(list(topology.nodes.values()), topology.connections)
        state = ExecutionState(self.exec_manager, job_id, len(image_ids))
        
        # Initialize captions for all images
        captions = {img_id: '' for img_id in image_ids}
        
        # Start from Input node and follow what's connected
        downstream = topology.get_downstream_nodes(input_node['id'])
        if not downstream:
            raise ValueError("Nothing connected to Input node")
        
        # Process all nodes connected from Input
        for target_id, from_port, to_port in downstream:
            if from_port != 0:  # Only follow main output (port 0) from Input
                continue
            
            target_node = topology.nodes.get(target_id)
            if not target_node:
                continue
            
            await self._execute_node_and_downstream(
                target_node, captions, image_ids, topology, output_node,
                prompt_resolver, get_model_func, set(), state
            )
    
    async def _execute_node_and_downstream(
        self, node: Dict, captions: Dict[str, str], image_ids: List[str],
        topology: GraphTopology, output_node: Dict, prompt_resolver: PromptResolver,
        get_model_func, visited: Set[str], state: ExecutionState
    ) -> None:
        """
        Execute a node and follow its outputs downstream.
        
        This recursively processes the graph by following connections.
        """
        if self.should_cancel or node['id'] in visited:
            return
        
        visited.add(node['id'])
        node_type = node['type']
        
        logger.info(f"Executing node: {node_type} ({node['id']})")
        
        # Execute based on node type
        if node_type == 'aimodel':
            await self._execute_ai_model(
                node, captions, image_ids, topology, output_node,
                prompt_resolver, get_model_func, state
            )
        elif node_type == 'curate':
            await self._execute_curate(
                node, captions, image_ids, topology, output_node,
                prompt_resolver, get_model_func, visited, state
            )
            return  # Curate handles its own downstream routing
        elif node_type == 'output':
            # Save captions that reached Output
            await self._save_final_captions(captions, image_ids, node, prompt_resolver)
            return  # Output is terminal
        elif node_type in ['prompt', 'conjunction']:
            # These don't execute, just pass through
            pass
        
        # Follow downstream connections
        for target_id, _, _ in topology.get_downstream_nodes(node['id']):
            target_node = topology.nodes.get(target_id)
            if target_node:
                await self._execute_node_and_downstream(
                    target_node, captions, image_ids, topology, output_node,
                    prompt_resolver, get_model_func, visited, state
                )
    
    async def _execute_ai_model(
        self, node: Dict, captions: Dict[str, str], image_ids: List[str],
        topology: GraphTopology, output_node: Dict, prompt_resolver: PromptResolver,
        get_model_func, state: ExecutionState
    ) -> None:
        """Execute an AI model node - just generate captions, don't save."""
        model_name = node['data'].get('model', 'blip')
        parameters = node['data'].get('parameters', {}).copy()
        batch_size = parameters.get('batch_size', 1)
        
        # Get static prompt
        base_prompt = prompt_resolver.get_static_prompt(node)
        
        # Get precision params
        from config import PRECISION_DEFAULTS
        precision_params = None
        
        # Check if precision or flash attention is specified in parameters
        if 'precision' in parameters or 'use_flash_attention' in parameters:
            # Use PRECISION_DEFAULTS as fallback, or model config defaults
            if model_name in PRECISION_DEFAULTS:
                defaults = PRECISION_DEFAULTS[model_name]
                precision_params = {
                    'precision': parameters.get('precision', defaults['precision']),
                    'use_flash_attention': parameters.get('use_flash_attention', 
                                                         defaults['use_flash_attention'])
                }
            else:
                # Model not in PRECISION_DEFAULTS, use only what's in parameters
                precision_params = {}
                if 'precision' in parameters:
                    precision_params['precision'] = parameters['precision']
                if 'use_flash_attention' in parameters:
                    precision_params['use_flash_attention'] = parameters['use_flash_attention']
        
        # Load model
        model_adapter = get_model_func(model_name, precision_params)
        if not model_adapter or not model_adapter.is_loaded():
            raise ValueError(f"Model {model_name} not available")
        
        # Process in batches
        prev_captions = [captions.get(img_id, '') for img_id in image_ids]
        
        for batch_start in range(0, len(image_ids), batch_size):
            batch_end = min(batch_start + batch_size, len(image_ids))
            batch_ids = image_ids[batch_start:batch_end]
            
            try:
                # Load images
                image_paths = await self.async_session.get_image_paths_batch(batch_ids)
                images = []
                valid_indices = []
                
                for i, img_id in enumerate(batch_ids):
                    img_path = image_paths.get(img_id)
                    if img_path:
                        images.append(load_image(img_path))
                        valid_indices.append(batch_start + i)
                
                if not images:
                    for _ in batch_ids:
                        state.increment(success=False)
                    continue
                
                # Build prompts
                prompts = [
                    prompt_resolver.get_prompt_for_image(node, prev_captions, idx, base_prompt)
                    for idx in valid_indices
                ]
                
                # Generate captions
                if len(images) > 1 and batch_size > 1:
                    new_captions = model_adapter.generate_captions_batch(images, prompts, parameters)
                else:
                    new_captions = [model_adapter.generate_caption(images[0], prompts[0], parameters)]
                
                # Update captions dict and state
                for idx, caption in zip(valid_indices, new_captions):
                    captions[image_ids[idx]] = caption
                    state.increment(success=True)
                
                # Periodic status update
                if batch_start % (batch_size * 1) == 0:  # Update every batch
                    state.update()
                    
            except Exception as e:
                logger.error(f"Batch error: {e}")
                for _ in batch_ids:
                    state.increment(success=False)
        
        # Final update after node
        state.update()
    
    async def _execute_curate(
        self, node: Dict, captions: Dict[str, str], image_ids: List[str],
        topology: GraphTopology, output_node: Dict, prompt_resolver: PromptResolver,
        get_model_func, visited: Set[str], state: ExecutionState
    ) -> None:
        """Execute curate node - routes images to different paths."""
        model_name = node['data'].get('model', 'blip2-opt-2.7b')
        parameters = node['data'].get('parameters', {}).copy()
        ports = node['data'].get('ports', [])
        template = node['data'].get('template', '')
        
        if template:
            parameters['template'] = template
        
        # TODO: Curate node feature is currently disabled due to missing VLMRouterAdapter
        # This needs to be reimplemented using the ModelAdapterFactory and a VLM model
        # to route images based on their content/captions to different ports.
        # For now, route all images to the default port.
        
        logger.warning("Curate node functionality is not implemented - routing all to default port")
        
        # Find default port or use first port
        default_port = next((p['id'] for p in ports if p.get('isDefault')), ports[0]['id'] if ports else None)
        
        if default_port is None:
            logger.error("Curate node has no ports configured")
            for _ in image_ids:
                state.increment(success=False)
            return
        
        # Route all images to default port
        routing = {port['id']: [] for port in ports}
        routing[default_port] = list(range(len(image_ids)))
        
        # Mark all as successful (since we're just passing through)
        for _ in image_ids:
            state.increment(success=True)
        
        # Update after routing
        state.update()
        
        # Follow each port's downstream path
        for port_index, port in enumerate(ports):
            port_id = port['id']
            routed_indices = routing.get(port_id, [])
            if not routed_indices:
                continue
            
            # Get images routed to this port
            port_image_ids = [image_ids[idx] for idx in routed_indices]
            port_captions = {img_id: captions[img_id] for img_id in port_image_ids}
            
            # Follow this port's downstream connections
            for target_id, from_port, to_port in topology.get_downstream_nodes(node['id']):
                if from_port != port_index:
                    continue
                
                target_node = topology.nodes.get(target_id)
                if target_node:
                    await self._execute_node_and_downstream(
                        target_node, port_captions, port_image_ids, topology, output_node,
                        prompt_resolver, get_model_func, visited.copy(), state
                    )
            
            # Update main captions dict
            captions.update(port_captions)
    
    async def _save_final_captions(
        self, captions: Dict[str, str], image_ids: List[str],
        output_node: Dict, prompt_resolver: PromptResolver
    ) -> None:
        """Save captions that reached the Output node."""
        for img_id in image_ids:
            caption = captions.get(img_id, '')
            if caption:  # Only save non-empty captions
                await self.async_session.save_caption(img_id, caption)
        
        logger.info(f"Saved {len([c for c in captions.values() if c])} captions to database")
