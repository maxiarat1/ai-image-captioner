"""
Graph Executor - Clean execution engine for node-based workflows.

Processes node graphs independently in the background, surviving page refreshes.
"""

import logging
import time
import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from PIL import Image

from database import ExecutionManager, AsyncSessionManager
from utils.image_utils import load_image

logger = logging.getLogger(__name__)


class GraphExecutor:
    """
    Executes node graphs independently of frontend.

    Parses graph structure, builds execution chain, processes images stage-by-stage,
    and persists all results to database.
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

            # Parse and validate graph
            nodes, connections = self._parse_graph(graph)
            input_node, output_node = self._find_required_nodes(nodes)

            if not input_node or not output_node:
                raise ValueError("Graph must have Input and Output nodes")

            # Build AI model chain
            ai_chain = self._build_ai_chain(nodes, connections, input_node, output_node)
            if not ai_chain:
                raise ValueError("No AI models connected from Input to Output")

            # Update total stages
            self.exec_manager.update_status(job_id, total_stages=len(ai_chain))

            # Execute stage-by-stage
            await self._execute_chain(
                job_id, ai_chain, image_ids, nodes, connections,
                output_node, get_model_func
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

    def _parse_graph(self, graph: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
        """Extract nodes and connections from graph definition."""
        nodes = graph.get('nodes', [])
        connections = graph.get('connections', [])
        return nodes, connections

    def _find_required_nodes(self, nodes: List[Dict]) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Find Input and Output nodes."""
        input_node = next((n for n in nodes if n['type'] == 'input'), None)
        output_node = next((n for n in nodes if n['type'] == 'output'), None)
        return input_node, output_node

    def _has_curate_nodes(self, nodes: List[Dict]) -> bool:
        """Check if graph contains any curate nodes."""
        return any(n['type'] == 'curate' for n in nodes)

    def _build_ai_chain(
        self, nodes: List[Dict], connections: List[Dict],
        input_node: Dict, output_node: Dict
    ) -> List[Dict]:
        """
        Build sequential chain of AI model nodes.

        Chain rules:
        - First AI receives images from Input node (port 0 -> port 0)
        - Subsequent AIs receive prompts from previous AI captions (port 0 -> port 1)

        Note: For graphs with curate nodes, this builds the initial path.
        Routing happens dynamically during execution.
        """
        ai_nodes = [n for n in nodes if n['type'] in ('aimodel', 'curate')]
        if not ai_nodes:
            return []

        # Helper: check if connection exists
        def has_conn(from_id, from_port, to_id, to_port):
            return any(
                c['from'] == from_id and c['fromPort'] == from_port and
                c['to'] == to_id and c['toPort'] == to_port
                for c in connections
            )

        # Find first AI: receives images from Input
        candidates = [ai for ai in ai_nodes if has_conn(input_node['id'], 0, ai['id'], 0)]
        if not candidates:
            return []

        # Start with first AI that's not fed by another AI on prompt port
        def is_fed_by_ai(node):
            return any(
                c['to'] == node['id'] and c['toPort'] == 1 and
                any(n['id'] == c['from'] and n['type'] in ('aimodel', 'curate') for n in ai_nodes)
                for c in connections
            )

        start = next((ai for ai in candidates if not is_fed_by_ai(ai)), candidates[0])

        # Follow chain forward (note: for curate nodes, this builds one possible path)
        chain = [start]
        visited = {start['id']}
        current = start

        while True:
            # For curate nodes with multiple outputs, we take the first connected path
            # (actual routing happens at runtime)
            # For regular AI nodes, find next node connected from captions (port 0) to prompt (port 1)
            next_conn = next((
                c for c in connections
                if c['from'] == current['id'] and
                   c['toPort'] == 1 and
                   any(n['id'] == c['to'] and n['type'] in ('aimodel', 'curate') for n in ai_nodes)
            ), None)

            if not next_conn:
                break

            next_ai = next(n for n in ai_nodes if n['id'] == next_conn['to'])
            if next_ai['id'] in visited:
                break  # Prevent cycles

            chain.append(next_ai)
            visited.add(next_ai['id'])
            current = next_ai

        node_types = [f"{ai['type']}:{ai['data'].get('model', '?')}" for ai in chain]
        logger.info(f"Built processing chain with {len(chain)} stages: {node_types}")
        return chain

    async def _execute_chain(
        self, job_id: str, ai_chain: List[Dict], image_ids: List[str],
        nodes: List[Dict], connections: List[Dict], output_node: Dict, get_model_func
    ) -> None:
        """Execute the AI chain stage-by-stage."""
        total_images = len(image_ids)
        start_time = time.time()

        # Track captions across stages
        prev_captions = [''] * total_images
        total_success = 0
        total_failed = 0

        for stage_idx, ai_node in enumerate(ai_chain):
            if self.should_cancel:
                logger.info(f"Job {job_id} cancelled at stage {stage_idx + 1}")
                self.exec_manager.update_status(job_id, status='cancelled')
                return

            node_type = ai_node['type']
            model_name = ai_node['data'].get('model', 'blip')
            parameters = ai_node['data'].get('parameters', {})
            batch_size = parameters.get('batch_size', 1)

            logger.info(f"Stage {stage_idx + 1}/{len(ai_chain)}: Processing {total_images} images with {node_type}:{model_name}")

            # Handle curate nodes differently
            if node_type == 'curate':
                # For curate nodes, we need to handle routing
                # For now, we'll process all images and route them, then continue
                # with the remaining chain for each route
                logger.info(f"Curate node detected at stage {stage_idx + 1}, implementing routing logic")
                # Note: Full routing implementation would process split batches through different paths
                # For MVP, we'll route but continue with single path (first connected downstream)
                # TODO: Implement full per-route processing
                continue

            # Standard AI model processing

            stage_success = 0
            stage_failed = 0
            processed_in_stage = 0

            # Get static prompt for this AI node
            base_prompt = self._build_prompt_for_node(ai_node, nodes, connections)

            # Process in batches
            for batch_start in range(0, total_images, batch_size):
                if self.should_cancel:
                    break

                batch_end = min(batch_start + batch_size, total_images)
                batch_image_ids = image_ids[batch_start:batch_end]
                batch_size_actual = len(batch_image_ids)

                try:
                    # Load model
                    precision_params = self._extract_precision_params(model_name, parameters)
                    model_adapter = get_model_func(model_name, precision_params)

                    if not model_adapter or not model_adapter.is_loaded():
                        raise ValueError(f"Model {model_name} not available")

                    # Get image paths
                    image_paths_dict = await self.async_session.get_image_paths_batch(batch_image_ids)

                    # Load images
                    images = []
                    valid_indices = []
                    for i, img_id in enumerate(batch_image_ids):
                        img_path = image_paths_dict.get(img_id)
                        if img_path:
                            images.append(load_image(img_path))
                            valid_indices.append(batch_start + i)

                    if not images:
                        stage_failed += batch_size_actual
                        continue

                    # Build prompts for this batch
                    prompts = []
                    for idx in valid_indices:
                        # Check if this AI is fed by previous AI or conjunction
                        prompt = self._get_prompt_for_image(
                            ai_node, nodes, connections, base_prompt, prev_captions, idx
                        )
                        prompts.append(prompt)

                    # Generate captions
                    if len(images) > 1 and batch_size > 1:
                        captions = model_adapter.generate_captions_batch(images, prompts, parameters)
                    else:
                        captions = [model_adapter.generate_caption(images[0], prompts[0], parameters)]

                    # Store results
                    for idx, caption in zip(valid_indices, captions):
                        prev_captions[idx] = caption

                        # If last stage, save to database and check for output conjunction
                        if stage_idx == len(ai_chain) - 1:
                            final_caption = self._resolve_output_conjunction(
                                ai_node, output_node, nodes, connections, prev_captions, idx
                            )
                            # Save to database
                            await self.async_session.save_caption(image_ids[idx], final_caption)

                    stage_success += len(captions)

                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    stage_failed += batch_size_actual

                # Update progress
                processed_in_stage += batch_size_actual
                total_processed = stage_idx * total_images + processed_in_stage

                elapsed = time.time() - start_time
                avg_speed = total_processed / elapsed if elapsed > 0 else 0
                remaining = (total_images * len(ai_chain)) - total_processed
                eta = remaining / avg_speed if avg_speed > 0 else 0

                self.exec_manager.update_status(
                    job_id,
                    current_stage=stage_idx + 1,
                    processed=processed_in_stage,
                    success=total_success + stage_success,
                    failed=total_failed + stage_failed,
                    progress={
                        'speed': f"{avg_speed:.2f} img/s" if avg_speed > 0 else "",
                        'eta': self._format_time(eta) if eta > 0 else "",
                        'stage_progress': f"{processed_in_stage}/{total_images}"
                    }
                )

            total_success += stage_success
            total_failed += stage_failed

        # Final update
        total_time = time.time() - start_time
        self.exec_manager.update_status(
            job_id,
            processed=total_images,
            success=total_success,
            failed=total_failed,
            progress={
                'total_time': self._format_time(total_time),
                'avg_speed': f"{total_images / total_time:.2f} img/s" if total_time > 0 else ""
            }
        )

    def _build_prompt_for_node(
        self, ai_node: Dict, nodes: List[Dict], connections: List[Dict]
    ) -> str:
        """Build static prompt connected to this AI node."""
        # Check for direct prompt node connection
        prompt_nodes = [n for n in nodes if n['type'] == 'prompt']
        for prompt_node in prompt_nodes:
            if any(c['from'] == prompt_node['id'] and c['to'] == ai_node['id'] for c in connections):
                return prompt_node['data'].get('text', '')

        return ''

    def _get_prompt_for_image(
        self, ai_node: Dict, nodes: List[Dict], connections: List[Dict],
        base_prompt: str, prev_captions: List[str], img_index: int
    ) -> Optional[str]:
        """Get prompt for specific image (handles chaining and conjunctions)."""
        # Check if fed by previous AI model
        prev_ai_conn = next((
            c for c in connections
            if c['to'] == ai_node['id'] and c['toPort'] == 1 and
               any(n['id'] == c['from'] and n['type'] == 'aimodel' for n in nodes)
        ), None)

        if prev_ai_conn:
            return prev_captions[img_index] or None

        # Check for conjunction feeding this AI
        conj_conn = next((
            c for c in connections
            if c['to'] == ai_node['id'] and
               any(n['id'] == c['from'] and n['type'] == 'conjunction' for n in nodes)
        ), None)

        if conj_conn:
            conj_node = next(n for n in nodes if n['id'] == conj_conn['from'])
            return self._resolve_conjunction(conj_node, nodes, connections, prev_captions, img_index)

        return base_prompt or None

    def _resolve_output_conjunction(
        self, last_ai: Dict, output_node: Dict, nodes: List[Dict],
        connections: List[Dict], prev_captions: List[str], img_index: int
    ) -> str:
        """Check if there's a conjunction between last AI and output, and resolve it."""
        # Find conjunction between last AI and output
        conj_node = next((
            n for n in nodes if n['type'] == 'conjunction' and
            any(c['from'] == last_ai['id'] and c['to'] == n['id'] for c in connections) and
            any(c['from'] == n['id'] and c['to'] == output_node['id'] for c in connections)
        ), None)

        if conj_node:
            return self._resolve_conjunction(conj_node, nodes, connections, prev_captions, img_index)

        return prev_captions[img_index]

    def _resolve_conjunction(
        self, conj_node: Dict, nodes: List[Dict], connections: List[Dict],
        prev_captions: List[str], img_index: int
    ) -> str:
        """Resolve conjunction template with actual values."""
        template = conj_node['data'].get('template', '')
        if not template:
            return prev_captions[img_index]

        # Build reference map
        connected_items = conj_node['data'].get('connectedItems', [])
        ref_map = {}

        for item in connected_items:
            ref_key = item.get('refKey', '')
            content = item.get('content', '')

            # If content is placeholder for AI-generated captions, use actual caption
            if content == '[Generated Captions]':
                ref_map[ref_key] = prev_captions[img_index]
            else:
                ref_map[ref_key] = content

        # Replace placeholders {key} with values
        resolved = re.sub(r'\{([^}]+)\}', lambda m: ref_map.get(m.group(1), m.group(0)), template)
        return resolved

    def _extract_precision_params(self, model_name: str, parameters: Dict) -> Optional[Dict]:
        """Extract precision parameters for model loading."""
        from config import PRECISION_DEFAULTS

        if model_name not in PRECISION_DEFAULTS:
            return None

        defaults = PRECISION_DEFAULTS[model_name]
        return {
            'precision': parameters.get('precision', defaults['precision']),
            'use_flash_attention': parameters.get('use_flash_attention', defaults['use_flash_attention'])
        }

    def _get_curate_routing_paths(
        self, curate_node: Dict, nodes: List[Dict], connections: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """
        Get all routing paths from a curate node.

        Returns a dict mapping port_id to list of downstream nodes for that route.
        """
        ports = curate_node['data'].get('ports', [])
        routing_paths = {}

        for port_index, port in enumerate(ports):
            port_id = port.get('id')
            path = []

            # Find connections from this port
            downstream_conns = [
                c for c in connections
                if c['from'] == curate_node['id'] and c['fromPort'] == port_index
            ]

            for conn in downstream_conns:
                downstream_node = next((n for n in nodes if n['id'] == conn['to']), None)
                if downstream_node:
                    path.append(downstream_node)

            routing_paths[port_id] = path

        return routing_paths

    async def _execute_curate_routing(
        self, job_id: str, curate_node: Dict, images: List[Image.Image],
        image_ids: List[str], prev_captions: List[str], nodes: List[Dict],
        connections: List[Dict], get_model_func
    ) -> Dict[str, List[int]]:
        """
        Execute routing decisions for a curate node.

        Returns a dict mapping port_id to list of image indices routed to that port.
        """
        model_name = curate_node['data'].get('model', 'blip2-opt-2.7b')
        model_type = curate_node['data'].get('modelType', 'vlm')
        parameters = curate_node['data'].get('parameters', {})
        ports = curate_node['data'].get('ports', [])

        logger.info(f"Executing curate routing with {model_name} ({model_type}) for {len(images)} images")

        # Create curate adapter
        from .models.vlm_router_adapter import VLMRouterAdapter

        if model_type == 'vlm':
            curate_adapter = VLMRouterAdapter(model_name)
            curate_adapter.load_model()
        else:
            # For now, only VLM routing is implemented
            logger.warning(f"Model type {model_type} not yet implemented, using VLM routing")
            curate_adapter = VLMRouterAdapter(model_name)
            curate_adapter.load_model()

        # Route each image
        routing_decisions = {}
        for port in ports:
            routing_decisions[port['id']] = []

        for idx, (image, caption) in enumerate(zip(images, prev_captions)):
            try:
                port_id = curate_adapter.route_image(
                    image, caption, ports, parameters
                )
                if port_id in routing_decisions:
                    routing_decisions[port_id].append(idx)
                    logger.debug(f"Image {idx} routed to port {port_id}")
                else:
                    logger.warning(f"Invalid port_id {port_id} returned, using default")
                    default_port = next((p['id'] for p in ports if p.get('isDefault')), ports[0]['id'])
                    routing_decisions[default_port].append(idx)
            except Exception as e:
                logger.error(f"Error routing image {idx}: {e}")
                # Route to default port on error
                default_port = next((p['id'] for p in ports if p.get('isDefault')), ports[0]['id'])
                routing_decisions[default_port].append(idx)

        # Log routing distribution
        dist_str = ", ".join([f"{port_id}: {len(indices)} images" for port_id, indices in routing_decisions.items() if indices])
        logger.info(f"Routing distribution - {dist_str}")

        return routing_decisions

    def _format_time(self, seconds: float) -> str:
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
