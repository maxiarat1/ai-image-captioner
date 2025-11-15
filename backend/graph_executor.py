"""
Graph Executor - Clean execution engine for node-based workflows.

Simplified architecture with clear separation of concerns:
- GraphParser: validates graph structure
- ChainBuilder: constructs execution sequence
- PromptResolver: handles all prompt/conjunction logic
- ProgressTracker: unified progress updates
- NodeExecutors: handle node-specific processing
"""

import asyncio
import logging
import time
import re
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image

from database import ExecutionManager, AsyncSessionManager
from utils.image_utils import load_image

logger = logging.getLogger(__name__)


# ============================================================================
# Helper Classes - Single Responsibility Components
# ============================================================================

class GraphParser:
    """Validates and extracts graph structure."""
    
    @staticmethod
    def parse(graph: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
        """Extract and validate nodes and connections."""
        nodes = graph.get('nodes', [])
        connections = graph.get('connections', [])
        return nodes, connections
    
    @staticmethod
    def find_io_nodes(nodes: List[Dict]) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Find required Input and Output nodes."""
        input_node = next((n for n in nodes if n['type'] == 'input'), None)
        output_node = next((n for n in nodes if n['type'] == 'output'), None)
        return input_node, output_node


class ChainBuilder:
    """Constructs execution chain from graph topology."""
    
    @staticmethod
    def build(nodes: List[Dict], connections: List[Dict], 
              input_node: Dict, output_node: Dict) -> List[Dict]:
        """
        Build sequential chain of AI/curate nodes.
        
        Chain starts from Input node (port 0) and follows connections
        through AI/curate nodes. Stops at cycles or dead ends.
        """
        ai_nodes = [n for n in nodes if n['type'] in ('aimodel', 'curate')]
        if not ai_nodes:
            return []
        
        # Find first AI connected to Input
        first_ai = ChainBuilder._find_first_node(ai_nodes, connections, input_node)
        if not first_ai:
            return []
        
        # Follow chain forward
        chain = [first_ai]
        visited = {first_ai['id']}
        current = first_ai
        
        while True:
            next_node = ChainBuilder._find_next_node(current, ai_nodes, connections, visited)
            if not next_node:
                break
            chain.append(next_node)
            visited.add(next_node['id'])
            current = next_node
        
        logger.info(f"Built chain with {len(chain)} stages: {[n['type'] for n in chain]}")
        return chain
    
    @staticmethod
    def _find_first_node(ai_nodes: List[Dict], connections: List[Dict], 
                         input_node: Dict) -> Optional[Dict]:
        """Find first AI node connected to Input."""
        for node in ai_nodes:
            if any(c['from'] == input_node['id'] and c['to'] == node['id'] 
                   for c in connections):
                return node
        return None
    
    @staticmethod
    def _find_next_node(current: Dict, ai_nodes: List[Dict], 
                       connections: List[Dict], visited: set) -> Optional[Dict]:
        """Find next AI node in chain."""
        for conn in connections:
            if conn['from'] == current['id']:
                next_node = next((n for n in ai_nodes if n['id'] == conn['to']), None)
                if next_node and next_node['id'] not in visited:
                    return next_node
        return None


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


class ProgressTracker:
    """Unified progress tracking and updates."""
    
    def __init__(self, exec_manager: ExecutionManager, job_id: str, 
                 total_images: int, total_stages: int):
        self.exec_manager = exec_manager
        self.job_id = job_id
        self.total_images = total_images
        self.total_stages = total_stages
        self.start_time = time.time()
        self.total_success = 0
        self.total_failed = 0
    
    def update_progress(self, stage_idx: int, processed_in_stage: int, 
                       stage_success: int, stage_failed: int):
        """Send progress update to database."""
        elapsed = time.time() - self.start_time
        total_processed = stage_idx * self.total_images + processed_in_stage
        
        avg_speed = total_processed / elapsed if elapsed > 0 else 0
        remaining = (self.total_images * self.total_stages) - total_processed
        eta = remaining / avg_speed if avg_speed > 0 else 0
        
        self.exec_manager.update_status(
            self.job_id,
            current_stage=stage_idx + 1,
            processed=processed_in_stage,
            success=self.total_success + stage_success,
            failed=self.total_failed + stage_failed,
            progress={
                'speed': f"{avg_speed:.2f} img/s" if avg_speed > 0 else "",
                'eta': self._format_time(eta) if eta > 0 else "",
                'stage_progress': f"{processed_in_stage}/{self.total_images}"
            }
        )
    
    def finalize_stage(self, stage_idx: int, stage_success: int, stage_failed: int):
        """Mark stage as complete."""
        self.total_success += stage_success
        self.total_failed += stage_failed
        
        self.exec_manager.update_status(
            self.job_id,
            current_stage=stage_idx + 1,
            processed=self.total_images,
            success=self.total_success,
            failed=self.total_failed,
            progress={
                'speed': "",
                'eta': "",
                'stage_progress': f"{self.total_images}/{self.total_images}"
            }
        )
    
    def finalize_job(self):
        """Mark job as complete with final stats."""
        total_time = time.time() - self.start_time
        avg_speed = self.total_images / total_time if total_time > 0 else 0
        
        self.exec_manager.update_status(
            self.job_id,
            processed=self.total_images,
            success=self.total_success,
            failed=self.total_failed,
            progress={
                'total_time': self._format_time(total_time),
                'avg_speed': f"{avg_speed:.2f} img/s" if avg_speed > 0 else ""
            }
        )
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time in seconds to human-readable string."""
        if seconds < 60:
            return f"~{int(seconds)}s"
        elif seconds < 3600:
            mins = int(seconds / 60)
            secs = int(seconds % 60)
            return f"~{mins}m {secs}s"
        else:
            hours = int(seconds / 3600)
            mins = int((seconds % 3600) / 60)
            return f"~{hours}h {mins}m"


# ============================================================================
# Node Executors - Handle Specific Node Types
# ============================================================================

class AIModelExecutor:
    """Executes AI model nodes."""
    
    def __init__(self, async_session: AsyncSessionManager, prompt_resolver: PromptResolver):
        self.async_session = async_session
        self.prompt_resolver = prompt_resolver
    
    async def execute(self, ai_node: Dict, image_ids: List[str], 
                     prev_captions: List[str], get_model_func, 
                     is_last_stage: bool, output_node: Dict) -> Tuple[int, int]:
        """
        Execute AI model node.
        
        Returns (success_count, failed_count).
        """
        model_name = ai_node['data'].get('model', 'blip')
        parameters = ai_node['data'].get('parameters', {})
        batch_size = parameters.get('batch_size', 1)
        
        logger.info(f"Processing {len(image_ids)} images with {model_name}")
        
        stage_success = 0
        stage_failed = 0
        base_prompt = self.prompt_resolver.get_static_prompt(ai_node)
        
        # Process in batches
        for batch_start in range(0, len(image_ids), batch_size):
            batch_end = min(batch_start + batch_size, len(image_ids))
            batch_ids = image_ids[batch_start:batch_end]
            
            try:
                # Load model
                precision_params = self._extract_precision_params(model_name, parameters)
                model_adapter = get_model_func(model_name, precision_params)
                
                if not model_adapter or not model_adapter.is_loaded():
                    raise ValueError(f"Model {model_name} not available")
                
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
                    stage_failed += len(batch_ids)
                    continue
                
                # Build prompts
                prompts = [
                    self.prompt_resolver.get_prompt_for_image(
                        ai_node, prev_captions, idx, base_prompt
                    )
                    for idx in valid_indices
                ]
                
                # Generate captions
                if len(images) > 1 and batch_size > 1:
                    captions = model_adapter.generate_captions_batch(images, prompts, parameters)
                else:
                    captions = [model_adapter.generate_caption(images[0], prompts[0], parameters)]
                
                # Store results
                for idx, caption in zip(valid_indices, captions):
                    prev_captions[idx] = caption
                    
                    # Save to database if last stage
                    if is_last_stage:
                        final_caption = self.prompt_resolver.resolve_output(
                            ai_node, output_node, prev_captions, idx
                        )
                        await self.async_session.save_caption(image_ids[idx], final_caption)
                
                stage_success += len(captions)
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                stage_failed += len(batch_ids)
        
        return stage_success, stage_failed
    
    @staticmethod
    def _extract_precision_params(model_name: str, parameters: Dict) -> Optional[Dict]:
        """Extract precision parameters for model loading."""
        from config import PRECISION_DEFAULTS
        
        if model_name not in PRECISION_DEFAULTS:
            return None
        
        defaults = PRECISION_DEFAULTS[model_name]
        return {
            'precision': parameters.get('precision', defaults['precision']),
            'use_flash_attention': parameters.get('use_flash_attention', 
                                                 defaults['use_flash_attention'])
        }


class CurateExecutor:
    """Executes curate routing nodes."""
    
    def __init__(self, async_session: AsyncSessionManager, prompt_resolver: PromptResolver,
                 nodes: List[Dict], connections: List[Dict]):
        self.async_session = async_session
        self.prompt_resolver = prompt_resolver
        self.nodes = nodes
        self.connections = connections
    
    async def execute(self, curate_node: Dict, images: List[Image.Image],
                     image_ids: List[str], prev_captions: List[str],
                     get_model_func, output_node: Dict) -> Tuple[int, int]:
        """
        Execute curate routing node.
        
        Returns (success_count, failed_count).
        """
        model_name = curate_node['data'].get('model', 'blip2-opt-2.7b')
        model_type = curate_node['data'].get('modelType', 'vlm')
        parameters = curate_node['data'].get('parameters', {}).copy()
        ports = curate_node['data'].get('ports', [])
        template = curate_node['data'].get('template', '')
        
        if template:
            parameters['template'] = template
        
        logger.info(f"Routing {len(images)} images with {model_name}")
        
        # Load model
        precision_params = AIModelExecutor._extract_precision_params(model_name, parameters)
        from models.vlm_router_adapter import VLMRouterAdapter
        
        curate_adapter = VLMRouterAdapter(model_name)
        curate_adapter.load_model(precision_params, get_model_func)
        
        # Get routing paths to determine which connect to output
        routing_paths = self._get_routing_paths(curate_node)
        
        # Route each image
        routing_decisions = {port['id']: [] for port in ports}
        stage_success = 0
        stage_failed = 0
        
        for idx, (image, caption) in enumerate(zip(images, prev_captions)):
            try:
                port_id = curate_adapter.route_image(image, caption, ports, parameters)
                
                if port_id not in routing_decisions:
                    # Invalid port, use default
                    port_id = next((p['id'] for p in ports if p.get('isDefault')), ports[0]['id'])
                
                routing_decisions[port_id].append(idx)
                
                # Save immediately if this port connects to output
                if self._port_connects_to_output(port_id, routing_paths, output_node):
                    final_caption = self._resolve_output_for_port(
                        port_id, routing_paths, prev_captions, idx
                    )
                    await self.async_session.save_caption(image_ids[idx], final_caption)
                
                stage_success += 1
                
            except Exception as e:
                logger.error(f"Routing error for image {idx}: {e}")
                default_port = next((p['id'] for p in ports if p.get('isDefault')), ports[0]['id'])
                routing_decisions[default_port].append(idx)
                stage_failed += 1
        
        # Log distribution
        dist = ", ".join([f"{pid}: {len(indices)}" for pid, indices in routing_decisions.items() if indices])
        logger.info(f"Routing distribution - {dist}")
        
        return stage_success, stage_failed
    
    def _get_routing_paths(self, curate_node: Dict) -> Dict[str, List[Dict]]:
        """Get downstream nodes for each port using BFS."""
        ports = curate_node['data'].get('ports', [])
        routing_paths = {}
        
        for port_index, port in enumerate(ports):
            path = []
            visited = set()
            queue = []
            
            # Find initial connections from this port
            for conn in self.connections:
                if conn['from'] == curate_node['id'] and conn['fromPort'] == port_index:
                    downstream = next((n for n in self.nodes if n['id'] == conn['to']), None)
                    if downstream and downstream['id'] not in visited:
                        queue.append(downstream)
                        visited.add(downstream['id'])
                        path.append(downstream)
            
            # BFS traversal
            while queue:
                current = queue.pop(0)
                
                # Stop at terminal nodes
                if current['type'] in ['output', 'aimodel', 'curate']:
                    continue
                
                # Follow connections
                for conn in self.connections:
                    if conn['from'] == current['id']:
                        next_node = next((n for n in self.nodes if n['id'] == conn['to']), None)
                        if next_node and next_node['id'] not in visited:
                            queue.append(next_node)
                            visited.add(next_node['id'])
                            path.append(next_node)
            
            routing_paths[port['id']] = path
        
        return routing_paths
    
    def _port_connects_to_output(self, port_id: str, routing_paths: Dict, 
                                 output_node: Dict) -> bool:
        """Check if port routes to output."""
        downstream = routing_paths.get(port_id, [])
        return any(n['type'] == 'output' for n in downstream)
    
    def _resolve_output_for_port(self, port_id: str, routing_paths: Dict,
                                 prev_captions: List[str], img_index: int) -> str:
        """Resolve output caption for routed port (handles conjunctions)."""
        downstream = routing_paths.get(port_id, [])
        
        # Check for conjunction before output
        conj_node = next((n for n in downstream if n['type'] == 'conjunction'), None)
        if conj_node:
            return self.prompt_resolver._resolve_conjunction(conj_node, prev_captions, img_index)
        
        return prev_captions[img_index]


# ============================================================================
# Main Executor - Orchestrates Everything
# ============================================================================

class GraphExecutor:
    """
    Main orchestrator for graph execution.
    
    Coordinates parsing, chain building, and execution using specialized components.
    """
    
    def __init__(self, exec_manager: ExecutionManager, async_session: AsyncSessionManager):
        self.exec_manager = exec_manager
        self.async_session = async_session
        self.should_cancel = False
    
    async def execute(self, job_id: str, get_model_func) -> None:
        """
        Execute a job by ID.
        
        Args:
            job_id: Job identifier
            get_model_func: Function to get model adapter (from app.py)
        """
        job = self.exec_manager.get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return
        
        try:
            # Mark as running
            self.exec_manager.update_status(job_id, status='running')
            
            graph = job['graph']
            image_ids = job['image_ids']
            
            # Parse graph
            nodes, connections = GraphParser.parse(graph)
            input_node, output_node = GraphParser.find_io_nodes(nodes)
            
            if not input_node or not output_node:
                raise ValueError("Graph must have Input and Output nodes")
            
            # Build execution chain
            ai_chain = ChainBuilder.build(nodes, connections, input_node, output_node)
            if not ai_chain:
                raise ValueError("No AI models connected from Input to Output")
            
            # Initialize components
            prompt_resolver = PromptResolver(nodes, connections)
            progress_tracker = ProgressTracker(
                self.exec_manager, job_id, len(image_ids), len(ai_chain)
            )
            
            self.exec_manager.update_status(job_id, total_stages=len(ai_chain))
            
            # Execute chain
            await self._execute_chain(
                job_id, ai_chain, image_ids, nodes, connections,
                output_node, get_model_func, prompt_resolver, progress_tracker
            )
            
            # Mark as completed
            self.exec_manager.update_status(job_id, status='completed')
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.exception(f"Job {job_id} failed: {e}")
            self.exec_manager.update_status(job_id, status='failed', error=str(e))
    
    def cancel(self):
        """Signal cancellation of current execution."""
        self.should_cancel = True
    
    async def _execute_chain(
        self, job_id: str, ai_chain: List[Dict], image_ids: List[str],
        nodes: List[Dict], connections: List[Dict], output_node: Dict,
        get_model_func, prompt_resolver: PromptResolver, 
        progress_tracker: ProgressTracker
    ) -> None:
        """Execute the AI chain stage-by-stage."""
        prev_captions = [''] * len(image_ids)
        
        ai_executor = AIModelExecutor(self.async_session, prompt_resolver)
        curate_executor = CurateExecutor(
            self.async_session, prompt_resolver, nodes, connections
        )
        
        for stage_idx, node in enumerate(ai_chain):
            if self.should_cancel:
                logger.info(f"Job {job_id} cancelled at stage {stage_idx + 1}")
                self.exec_manager.update_status(job_id, status='cancelled')
                return
            
            node_type = node['type']
            is_last_stage = (stage_idx == len(ai_chain) - 1)
            
            logger.info(f"Stage {stage_idx + 1}/{len(ai_chain)}: {node_type}")
            
            # Execute node
            if node_type == 'curate':
                # Load all images for curate
                image_paths = await self.async_session.get_image_paths_batch(image_ids)
                images = []
                for img_id in image_ids:
                    img_path = image_paths.get(img_id)
                    if img_path:
                        images.append(load_image(img_path))
                    else:
                        logger.warning(f"Image path not found for {img_id}")
                        images.append(None)
                
                stage_success, stage_failed = await curate_executor.execute(
                    node, images, image_ids, prev_captions, get_model_func, output_node
                )
            else:
                # AI model execution
                stage_success, stage_failed = await ai_executor.execute(
                    node, image_ids, prev_captions, get_model_func, is_last_stage, output_node
                )
            
            # Update progress
            progress_tracker.finalize_stage(stage_idx, stage_success, stage_failed)
            
            # Small delay for frontend polling
            await asyncio.sleep(0.5)
        
        # Final stats
        progress_tracker.finalize_job()
