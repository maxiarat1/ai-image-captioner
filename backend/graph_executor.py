"""
Graph Executor - True graph-based execution engine for node workflows.

Architecture:
- GraphExecutor: orchestrates execution by following actual graph connections
- Nodes are executed only when their inputs are ready
- Data flows through connections naturally
- Only data reaching Output node is saved
"""

import asyncio
import logging
import time
import re
from typing import Dict, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass
from database import ExecutionManager
from utils.image_utils import load_image
from flow_control import FlowControlHub, create_processed_data

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

class NodePort:
    """
    Port index constants for node connections.

    These constants define standard port indices used throughout the graph execution:
    - Port 0 is typically the main/default port for most nodes
    - Port 1 on AI models is used for prompt input (enables caption chaining)
    """
    DEFAULT_OUTPUT = 0      # Main output port for most nodes
    IMAGE_INPUT = 0         # Image input port
    PROMPT_INPUT = 1        # Prompt/text input port (for caption chaining)


@dataclass
class ExecutionContext:
    """
    Context object for graph execution - replaces 9-parameter method signatures.

    This dataclass encapsulates all the data needed during graph traversal,
    making method signatures cleaner and easier to extend.
    """
    # Immutable context (set once at start)
    topology: 'GraphTopology'
    prompt_resolver: 'PromptResolver'
    state: 'ExecutionState'
    output_node: Dict
    get_model_func: Callable
    nodes_feeding_output: Set[str]  # Nodes that should save to DB

    # Mutable execution state (modified during execution)
    captions: Dict[str, str]
    image_ids: List[str]
    visited: Set[str]


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
            if to_port == NodePort.PROMPT_INPUT and source_type == 'aimodel':
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
        self.current_node_processed = 0  # Progress for current node
        self.success = 0  # Final success count (only from last node)
        self.failed = 0
        self.current_node_id = None  # Track which node is currently executing
    
    def set_current_node(self, node_id: str):
        """Set the currently executing node and reset its progress counter."""
        self.current_node_id = node_id
        self.current_node_processed = 0
    
    def increment_progress(self):
        """Increment progress for current node (for UI display)."""
        self.current_node_processed += 1
    
    def increment(self, success: bool = True):
        """Increment final counters (only called from final node)."""
        if success:
            self.success += 1
        else:
            self.failed += 1
    
    def update(self):
        """Send current state to database."""
        elapsed = time.time() - self.start_time
        speed = self.current_node_processed / elapsed if elapsed > 0 else 0
        
        self.exec_manager.update_status(
            self.job_id,
            current_stage=1,
            total_stages=1,
            processed=self.current_node_processed,  # Show current node's progress
            success=self.success,
            failed=self.failed,
            progress={
                'speed': f"{speed:.1f} img/s" if speed > 0 else "",
                'progress': f"{self.current_node_processed}/{self.total_images}",
                'current_node_id': self.current_node_id  # Send to frontend for highlighting
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
    5. Route data through Flow Control Hub to appropriate destinations
    """

    def __init__(self, exec_manager: ExecutionManager, flow_hub: FlowControlHub):
        self.exec_manager = exec_manager
        self.flow_hub = flow_hub
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

            # Wait 0.5s to ensure final status update reaches frontend via SSE
            await asyncio.sleep(0.5)

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

    
    def _find_nodes_on_path_to_output(
        self, output_id: str, topology: GraphTopology
    ) -> Set[str]:
        """
        Find all nodes that are on any path leading to the Output node.
        
        Uses backward traversal from Output to identify which nodes contribute
        to the final result. This prevents execution of dead-end branches.
        
        Args:
            output_id: ID of the output node
            topology: Graph topology
            
        Returns:
            Set of node IDs that are on a path to Output
        """
        nodes_on_path = set()
        queue = [output_id]
        nodes_on_path.add(output_id)
        
        while queue:
            current_id = queue.pop(0)
            
            # Get all nodes that feed into this node
            for source_id, _, _ in topology.get_upstream_nodes(current_id):
                if source_id not in nodes_on_path:
                    nodes_on_path.add(source_id)
                    queue.append(source_id)
        
        return nodes_on_path

    def _find_nodes_feeding_output(self, output_id: str, topology: GraphTopology) -> Set[str]:
        """
        Find nodes that DIRECTLY feed into the Output node.
        
        These are the nodes whose results should be saved to database,
        as they represent the final processed data.
        
        Args:
            output_id: ID of the output node
            topology: Graph topology
            
        Returns:
            Set of node IDs that directly connect to Output
        """
        nodes_feeding_output = set()
        
        # Get all nodes that directly feed into Output
        for source_id, _, _ in topology.get_upstream_nodes(output_id):
            # Skip non-processing nodes (prompt, conjunction)
            source_node = topology.nodes.get(source_id)
            if source_node and source_node['type'] in ['aimodel', 'curate']:
                nodes_feeding_output.add(source_id)
        
        return nodes_feeding_output
    
    def cancel(self):
        """Signal cancellation of current execution."""
        self.should_cancel = True

    def _resolve_precision_params(self, model_name: str, parameters: Dict) -> Optional[Dict]:
        """
        Resolve precision parameters for model loading.

        Merges user-specified parameters with model-specific defaults from config.

        Args:
            model_name: Name of the model
            parameters: User-specified parameters (may include precision, use_flash_attention)

        Returns:
            Dictionary with precision and flash_attention params, or None if not specified
        """
        # No precision params specified - return None to use model defaults
        if 'precision' not in parameters and 'use_flash_attention' not in parameters:
            return None

        # Get defaults if available for this model
        from app.models import get_factory
        factory = get_factory()
        defaults = factory.get_precision_defaults(model_name)

        if not defaults:
            defaults = {}

        # Merge user parameters with defaults
        return {
            'precision': parameters.get('precision', defaults.get('precision')),
            'use_flash_attention': parameters.get('use_flash_attention',
                                                 defaults.get('use_flash_attention'))
        }
    

    async def _execute_graph(
        self, job_id: str, image_ids: List[str], topology: GraphTopology,
        input_node: Dict, output_node: Dict, get_model_func
    ) -> None:
        """
        Execute graph by following connections from Input to Output.

        Uses queue-based iterative execution instead of recursion to:
        - Avoid stack overflow on deep graphs
        - Make execution flow clearer and easier to debug
        - Enable better control over execution order
        """
        # Build execution context
        prompt_resolver = PromptResolver(list(topology.nodes.values()), topology.connections)
        state = ExecutionState(self.exec_manager, job_id, len(image_ids))

        # Find all nodes that lead to Output (backward traversal)
        nodes_on_path = self._find_nodes_on_path_to_output(output_node['id'], topology)

        # Find nodes that DIRECTLY feed Output (these save to DB for dynamic updates)
        nodes_feeding_output = self._find_nodes_feeding_output(output_node['id'], topology)
        logger.info(f"Nodes on path to Output: {len(nodes_on_path)}, Nodes feeding Output: {len(nodes_feeding_output)}")

        ctx = ExecutionContext(
            topology=topology,
            prompt_resolver=prompt_resolver,
            state=state,
            output_node=output_node,
            get_model_func=get_model_func,
            nodes_feeding_output=nodes_feeding_output,
            captions={img_id: '' for img_id in image_ids},
            image_ids=image_ids,
            visited=set()
        )

        # Get nodes connected from Input
        downstream = topology.get_downstream_nodes(input_node['id'])
        if not downstream:
            raise ValueError("Nothing connected to Input node")

        # Queue-based execution (breadth-first traversal)
        from collections import deque
        queue = deque()

        # Add all nodes connected from Input node to queue (only if on path to Output)
        for target_id, from_port, to_port in downstream:
            if from_port == NodePort.DEFAULT_OUTPUT:  # Only follow main output from Input
                target_node = topology.nodes.get(target_id)
                if target_node and target_id in nodes_on_path:
                    queue.append(target_node)

        # Process queue until empty
        while queue and not self.should_cancel:
            node = queue.popleft()

            # Skip if already visited
            if node['id'] in ctx.visited:
                continue

            # Execute this node
            await self._execute_node(node, ctx)

            # Add downstream nodes to queue (unless curate or output handled it)
            if node['type'] not in ['curate', 'output']:
                for target_id, _, _ in topology.get_downstream_nodes(node['id']):
                    target_node = topology.nodes.get(target_id)
                    if target_node and target_node['id'] not in ctx.visited and target_id in nodes_on_path:
                        queue.append(target_node)

    async def _execute_node(self, node: Dict, ctx: ExecutionContext) -> None:
        """
        Execute a single node based on its type.

        Args:
            node: Node to execute
            ctx: Execution context with all necessary data
        """
        if self.should_cancel or node['id'] in ctx.visited:
            return

        ctx.visited.add(node['id'])
        node_type = node['type']

        logger.info(f"Executing node: {node_type} ({node['id']})")

        # Execute based on node type
        if node_type == 'aimodel':
            await self._execute_ai_model(node, ctx)
        elif node_type == 'curate':
            await self._execute_curate(node, ctx)
        elif node_type == 'output':
            await self._save_final_captions(node, ctx)
        elif node_type in ['prompt', 'conjunction']:
            # These don't execute, just pass through
            pass
    
    async def _load_batch_images(
        self, batch_ids: List[str], batch_start: int
    ) -> Tuple[List, List[int]]:
        """
        Load images for a batch of image IDs.

        Args:
            batch_ids: List of image IDs to load
            batch_start: Starting index in the full image list

        Returns:
            Tuple of (loaded images, valid indices in full list)
        """
        image_paths = await self.flow_hub.async_session.get_image_paths_batch(batch_ids)
        images = []
        valid_indices = []

        for i, img_id in enumerate(batch_ids):
            img_path = image_paths.get(img_id)
            if img_path:
                images.append(load_image(img_path))
                valid_indices.append(batch_start + i)

        return images, valid_indices

    def _build_batch_prompts(
        self, node: Dict, prev_captions: List[str], valid_indices: List[int],
        base_prompt: str, prompt_resolver: PromptResolver
    ) -> List[Optional[str]]:
        """
        Build prompts for a batch of images.

        Args:
            node: AI model node configuration
            prev_captions: Previous captions for all images
            valid_indices: Indices of successfully loaded images
            base_prompt: Base static prompt
            prompt_resolver: Resolver for dynamic prompts

        Returns:
            List of prompts (one per valid image)
        """
        return [
            prompt_resolver.get_prompt_for_image(node, prev_captions, idx, base_prompt)
            for idx in valid_indices
        ]

    async def _generate_batch_captions(
        self, images: List, prompts: List[Optional[str]], model_adapter,
        parameters: Dict, batch_size: int
    ) -> List[str]:
        """
        Generate captions for a batch of images.

        Uses batch processing if multiple images, otherwise single generation.

        Args:
            images: List of loaded PIL images
            prompts: List of prompts (one per image)
            model_adapter: Loaded model adapter
            parameters: Generation parameters
            batch_size: Configured batch size

        Returns:
            List of generated captions
        """
        if len(images) > 1 and batch_size > 1:
            return model_adapter.generate_captions_batch(images, prompts, parameters)
        else:
            return [model_adapter.generate_caption(images[0], prompts[0], parameters)]

    def _update_captions_and_state(
        self, valid_indices: List[int], new_captions: List[str],
        image_ids: List[str], captions: Dict[str, str], state: ExecutionState,
        is_final_node: bool = False
    ) -> None:
        """
        Update captions dictionary and execution state.

        Args:
            valid_indices: Indices of successfully processed images
            new_captions: Generated captions
            image_ids: Full list of image IDs
            captions: Captions dictionary to update
            state: Execution state to update
            is_final_node: Whether this is the final node (affects success counting)
        """
        for idx, caption in zip(valid_indices, new_captions):
            captions[image_ids[idx]] = caption
            
            # Always increment progress (for UI visibility during all nodes)
            state.increment_progress()
            
            # Only increment final success count from the last node
            if is_final_node:
                state.increment(success=True)

    async def _execute_ai_model(self, node: Dict, ctx: ExecutionContext) -> None:
        """
        Execute an AI model node - generate captions and update internal state.

        This method orchestrates the batch processing pipeline:
        1. Load model with precision parameters
        2. Process images in batches
        3. For each batch: load images → build prompts → generate captions
        4. Update state and captions dictionary

        Args:
            node: AI model node configuration
            ctx: Execution context with all necessary data
        """
        model_name = node['data'].get('model', 'blip')
        parameters = node['data'].get('parameters', {}).copy()
        batch_size = parameters.get('batch_size', 1)

        # Get static prompt and resolve precision parameters
        base_prompt = ctx.prompt_resolver.get_static_prompt(node)
        precision_params = self._resolve_precision_params(model_name, parameters)

        # Load model
        model_adapter = ctx.get_model_func(model_name, precision_params)
        if not model_adapter or not model_adapter.is_loaded():
            raise ValueError(f"Model {model_name} not available")

        # Prepare caption context for prompt resolution
        prev_captions = [ctx.captions.get(img_id, '') for img_id in ctx.image_ids]

        # Check if this is the final node (for state tracking)
        is_final_node = node['id'] in ctx.nodes_feeding_output

        # Set current node for progress tracking and UI highlighting
        ctx.state.set_current_node(node['id'])
        ctx.state.update()  # Send initial update to show which node started

        # Process in batches
        for batch_start in range(0, len(ctx.image_ids), batch_size):
            batch_end = min(batch_start + batch_size, len(ctx.image_ids))
            batch_ids = ctx.image_ids[batch_start:batch_end]

            try:
                # Load images for this batch
                images, valid_indices = await self._load_batch_images(batch_ids, batch_start)

                if not images:
                    # No valid images in this batch
                    for _ in batch_ids:
                        ctx.state.increment_progress()  # Show progress
                        if is_final_node:
                            ctx.state.increment(success=False)  # Count as failed in final stats
                    continue

                # Build prompts for valid images
                prompts = self._build_batch_prompts(
                    node, prev_captions, valid_indices, base_prompt, ctx.prompt_resolver
                )

                # Generate captions
                new_captions = await self._generate_batch_captions(
                    images, prompts, model_adapter, parameters, batch_size
                )

                # Update captions (always) and state (progress always, success only from final)
                self._update_captions_and_state(
                    valid_indices, new_captions, ctx.image_ids, ctx.captions, ctx.state,
                    is_final_node=is_final_node
                )

                # Save to database ONLY if this node directly feeds Output
                # This prevents intermediate results from polluting the Results tab
                # and eliminates redundant database writes
                if is_final_node:
                    processed_batch = [
                        create_processed_data(
                            image_id=ctx.image_ids[idx],
                            content=caption,
                            model_name=model_adapter.model_name,
                            parameters=parameters,
                            metadata={'node_id': node['id'], 'node_type': 'aimodel'},
                            sequence_num=idx
                        )
                        for idx, caption in zip(valid_indices, new_captions)
                    ]
                    await self.flow_hub.route_batch(processed_batch)

                # Periodic status update
                if batch_start % batch_size == 0:
                    ctx.state.update()

            except Exception as e:
                logger.error(f"Batch error: {e}")
                # Show progress for all nodes, count failures only from final
                for _ in batch_ids:
                    ctx.state.increment_progress()  # Show progress
                    if is_final_node:
                        ctx.state.increment(success=False)  # Count as failed in final stats

        # Final update after all batches
        ctx.state.update()
    
    async def _execute_curate(self, node: Dict, ctx: ExecutionContext) -> None:
        """
        Execute curate node - routes images to different paths.

        Note: Currently simplified to route all images through default port.
        Full VLM-based routing to be implemented when VLMRouterAdapter is available.

        Args:
            node: Curate node configuration
            ctx: Execution context
        """
        ports = node['data'].get('ports', [])

        # TODO: Curate node feature is currently disabled due to missing VLMRouterAdapter
        # This needs to be reimplemented using the ModelAdapterFactory and a VLM model
        # to route images based on their content/captions to different ports.
        # For now, route all images to the default port.

        logger.warning("Curate node functionality is not implemented - routing all to default port")

        # Find default port or use first port
        default_port = next((p['id'] for p in ports if p.get('isDefault')), ports[0]['id'] if ports else None)

        if default_port is None:
            logger.error("Curate node has no ports configured")
            for _ in ctx.image_ids:
                ctx.state.increment(success=False)
            return

        # Mark all as successful (since we're just passing through)
        for _ in ctx.image_ids:
            ctx.state.increment(success=True)

        # Update after routing
        ctx.state.update()

        # Note: With queue-based execution, downstream nodes will be added to queue
        # by the main execution loop, so we don't need to handle routing here
    
    async def _save_final_captions(self, node: Dict, ctx: ExecutionContext) -> None:
        """
        Route captions to database as a fallback.

        Only saves if no processing node directly fed Output. This handles cases
        where non-processing nodes (e.g., Conjunction) sit between the last AI model
        and Output.

        Args:
            node: Output node configuration
            ctx: Execution context with captions and image IDs
        """
        # If processing nodes directly fed Output, they already saved to DB
        # Only save as fallback if no processing nodes directly connected
        if ctx.nodes_feeding_output:
            logger.info(f"Skipping Output node save - {len(ctx.nodes_feeding_output)} processing nodes already saved to DB")
            return

        # Fallback: Save captions when only non-processing nodes feed Output
        logger.info("Output node saving captions (no direct processing nodes)")
        processed_batch = []

        for sequence_num, img_id in enumerate(ctx.image_ids):
            caption = ctx.captions.get(img_id, '')
            if caption:  # Only route non-empty captions
                processed_data = create_processed_data(
                    image_id=img_id,
                    content=caption,
                    model_name="graph_output",
                    parameters={},
                    metadata={'output_node_id': node['id']},
                    sequence_num=sequence_num
                )
                processed_batch.append(processed_data)

        # Route all data through the centralized flow control hub
        if processed_batch:
            routed_count = await self.flow_hub.route_batch(processed_batch)
            logger.info(f"Routed {routed_count}/{len(processed_batch)} captions through Flow Control Hub")
