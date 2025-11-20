"""
Registered node executors for the graph runtime.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from flow_control import create_processed_data
from utils.image_utils import load_image

from .context import GraphExecutionContext
from .constants import NodePort

logger = logging.getLogger(__name__)

DEFAULT_NODE_LABELS = {
    'aimodel': 'AI Model',
    'prompt': 'Prompt',
    'conjunction': 'Conjunction',
    'curate': 'Curate',
    'input': 'Input',
    'output': 'Output',
}


class BaseNodeExecutor:
    """Base class for all node executors with shared functionality."""
    node_type: str

    async def run(self, node: Dict, ctx: GraphExecutionContext) -> None:
        raise NotImplementedError

    async def _persist_terminal_output(
        self,
        ctx: GraphExecutionContext,
        node: Dict,
        outputs: Dict[str, str],
        model_name: str = "graph_output",
        parameters: Optional[Dict] = None,
        extra_metadata: Optional[Dict] = None
    ) -> None:
        """
        Generic method to persist output when this node is a terminal node.
        
        Call this immediately after processing to enable live streaming of results
        to the results tab. Any node connected to Output can use this method.
        
        Args:
            ctx: Execution context
            node: The current node
            outputs: Dict mapping image_id to output content
            model_name: Name to record for the model/source
            parameters: Optional parameters to record
            extra_metadata: Optional additional metadata to include
        """
        if not outputs:
            return
            
        processed_batch = []
        metadata_base = {
            'node_id': node['id'],
            'node_type': node.get('type', self.node_type)
        }
        if extra_metadata:
            metadata_base.update(extra_metadata)
        
        for img_id, content in outputs.items():
            if not content:
                continue
                
            sequence_num = ctx.image_ids.index(img_id) if img_id in ctx.image_ids else 0
            
            processed_batch.append(
                create_processed_data(
                    image_id=img_id,
                    content=str(content),
                    model_name=model_name,
                    parameters=parameters or {},
                    metadata=metadata_base.copy(),
                    sequence_num=sequence_num,
                    data_type='text'
                )
            )
            
            # Update state for live progress tracking
            ctx.state.increment(success=True)
        
        if processed_batch:
            await ctx.flow_hub.route_batch(processed_batch)


class InputNodeExecutor(BaseNodeExecutor):
    node_type = 'input'

    async def run(self, node: Dict, ctx: GraphExecutionContext) -> None:
        ctx.state.set_current_node(node['id'])
        values = {img_id: img_id for img_id in ctx.image_ids}
        ctx.buffers.set_output(node['id'], NodePort.DEFAULT_OUTPUT, values)
        ctx.state.current_node_processed = len(ctx.image_ids)
        ctx.state.update()


class PromptNodeExecutor(BaseNodeExecutor):
    node_type = 'prompt'

    async def run(self, node: Dict, ctx: GraphExecutionContext) -> None:
        ctx.state.set_current_node(node['id'])
        ctx.state.current_node_processed = len(ctx.image_ids)
        ctx.state.update()

        text = node['data'].get('text', '')
        outputs = {img_id: text for img_id in ctx.image_ids}
        ctx.buffers.set_output(node['id'], NodePort.DEFAULT_OUTPUT, outputs)

        # Persist immediately for live streaming when connected to Output
        if node['id'] in ctx.plan.terminal_nodes and text:
            await self._persist_terminal_output(
                ctx, node, outputs,
                model_name="prompt"
            )


class ConjunctionNodeExecutor(BaseNodeExecutor):
    node_type = 'conjunction'

    async def run(self, node: Dict, ctx: GraphExecutionContext) -> None:
        template = node['data'].get('template', '')
        connected_items = node['data'].get('connectedItems') or []

        ctx.state.set_current_node(node['id'])
        ctx.state.update()

        reference_values = self._build_reference_values(node, connected_items, ctx)
        resolved_outputs = self._resolve_outputs(node, template, reference_values, ctx)

        if not resolved_outputs:
            logger.warning("Conjunction node %s produced no outputs", node['id'])
            return

        ctx.buffers.set_output(node['id'], NodePort.DEFAULT_OUTPUT, resolved_outputs)

        is_terminal = node['id'] in ctx.plan.terminal_nodes

        # Process each output for live streaming
        for img_id in ctx.image_ids:
            content = resolved_outputs.get(img_id, '')
            ctx.state.increment_progress()
            
            # Persist immediately for live streaming when connected to Output
            if is_terminal and content:
                await self._persist_terminal_output(
                    ctx, node, {img_id: content},
                    model_name="conjunction",
                    extra_metadata={'template': template}
                )

        ctx.state.update()

    def _build_reference_values(
        self,
        node: Dict,
        connected_items: List[Dict],
        ctx: GraphExecutionContext
    ) -> Dict[str, Dict[str, str]]:
        values: Dict[str, Dict[str, str]] = {}

        for item in connected_items:
            ref_key = item.get('refKey')
            if not ref_key:
                continue
            values[ref_key] = self._get_item_values(item, ctx)

        # Ensure standard aliases for upstream nodes (e.g., {AI_Model})
        for conn in ctx.plan.upstream.get(node['id'], []):
            source_id = conn.source_id
            source_node = ctx.nodes_by_id.get(source_id)
            if not source_node:
                continue

            node_values = self._get_values_for_source(source_id, source_node, ctx)
            if not node_values:
                continue

            for ref_key in self._derive_ref_keys_for_node(source_node):
                values.setdefault(ref_key, node_values)

        return values

    def _get_item_values(self, item: Dict, ctx: GraphExecutionContext) -> Dict[str, str]:
        content = item.get('content', '')
        if content == '[Generated Captions]':
            return {img_id: ctx.captions.get(img_id, '') for img_id in ctx.image_ids}

        source_id = item.get('sourceId')
        if source_id:
            buffered = ctx.buffers.get_output(source_id, NodePort.DEFAULT_OUTPUT)
            if buffered:
                return {
                    img_id: str(buffered.get(img_id, '') or '')
                    for img_id in ctx.image_ids
                }

        if content:
            return {img_id: content for img_id in ctx.image_ids}

        return {img_id: '' for img_id in ctx.image_ids}

    def _get_values_for_source(
        self,
        source_id: str,
        source_node: Dict,
        ctx: GraphExecutionContext
    ) -> Optional[Dict[str, str]]:
        buffered = ctx.buffers.get_output(source_id, NodePort.DEFAULT_OUTPUT)
        if buffered:
            return {
                img_id: str(buffered.get(img_id, '') or '')
                for img_id in ctx.image_ids
            }

        if source_node.get('type') == 'aimodel':
            return {img_id: ctx.captions.get(img_id, '') for img_id in ctx.image_ids}

        return None

    def _derive_ref_keys_for_node(self, node: Dict) -> List[str]:
        candidates: List[str] = []

        label = node.get('label')
        if label:
            sanitized = self._sanitize_label(label)
            if sanitized:
                candidates.append(sanitized)

        default_label = DEFAULT_NODE_LABELS.get(node.get('type'))
        if default_label:
            sanitized = self._sanitize_label(default_label)
            if sanitized:
                candidates.append(sanitized)

        return list(dict.fromkeys(candidates))  # Preserves order, removes duplicates

    @staticmethod
    def _sanitize_label(label: str) -> str:
        if not label:
            return ""
        sanitized = re.sub(r'\s+', '_', label.strip())
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '', sanitized)
        return sanitized[:30]

    def _resolve_outputs(
        self,
        node: Dict,
        template: str,
        reference_values: Dict[str, Dict[str, str]],
        ctx: GraphExecutionContext
    ) -> Dict[str, str]:
        if not template.strip() and reference_values:
            # Empty template behaves like a passthrough of the first reference
            first_key = next(iter(reference_values))
            return reference_values[first_key]

        placeholder_pattern = re.compile(r'\{([^}]+)\}')
        missing_keys = set()
        outputs: Dict[str, str] = {}

        for img_id in ctx.image_ids:
            def replacer(match):
                key = match.group(1)
                value_map = reference_values.get(key)
                if not value_map:
                    missing_keys.add(key)
                    return ''
                return value_map.get(img_id, '')

            outputs[img_id] = placeholder_pattern.sub(replacer, template or '')

        if missing_keys:
            logger.debug(
                "Conjunction node %s missing values for references: %s",
                node['id'],
                sorted(missing_keys)
            )

        return outputs


class CurateNodeExecutor(BaseNodeExecutor):
    """
    Intelligent image routing node that uses VLM to categorize images
    and route them to different output ports based on evaluation.

    Design Principles:
    - Single Responsibility: Each method has one clear purpose
    - Fail-Fast Validation: Validate configuration before processing
    - Structured Output: Use JSON for reliable VLM responses
    - Explicit Routing: Clear decision-making with full observability
    - Graceful Degradation: Handle errors without breaking pipeline
    """
    node_type = 'curate'

    async def run(self, node: Dict, ctx: GraphExecutionContext) -> None:
        """Main execution flow with clear separation of concerns."""
        # 1. Validate and extract configuration
        config = self._validate_and_extract_config(node, ctx)
        if not config:
            return  # Validation failed, errors already logged

        # 2. Initialize execution state
        ctx.state.set_current_node(node['id'])
        ctx.state.update()

        # 3. Load model
        model_adapter = self._load_model(config['model_name'], config['parameters'], ctx)
        if not model_adapter:
            return

        # 4. Gather input captions
        input_captions = self._gather_input_captions(node, ctx)

        # 5. Build categorization prompt
        prompt = self._build_prompt(config)

        # 6. Process images and route to ports (includes live streaming persistence)
        port_outputs = await self._process_and_route(
            node, ctx, config, model_adapter, prompt, input_captions
        )

        # 7. Set outputs for downstream nodes
        await self._finalize_outputs(node, ctx, config, port_outputs)

        ctx.state.update()

    def _validate_and_extract_config(self, node: Dict, ctx: GraphExecutionContext) -> Optional[Dict]:
        """Validate node configuration and return extracted config or None."""
        model_type = node['data'].get('modelType', 'vlm')
        model_name = node['data'].get('model')
        ports = node['data'].get('ports', [])
        template = node['data'].get('template', '')
        parameters = node['data'].get('parameters', {}).copy()

        # Validation checks
        if not model_name:
            logger.error("Curate node %s: No model selected", node['id'])
            return None

        if not ports:
            logger.error("Curate node %s: No routing ports defined", node['id'])
            return None

        if model_type != 'vlm':
            logger.error("Curate node %s: Only VLM mode supported, got %s", node['id'], model_type)
            return None

        # Get connected ports
        downstream_conns = ctx.plan.downstream.get(node['id'], [])
        connected_indices = {conn.from_port for conn in downstream_conns}

        logger.debug("Curate node %s: connected_indices=%s, total_ports=%d",
                     node['id'], connected_indices, len(ports))

        active_ports = []
        port_index_map = {}

        for idx, port in enumerate(ports):
            port_label = port.get('label', f'Port {idx}')
            port_id = port.get('id', 'no-id')
            logger.debug("  Port idx=%d, label='%s', id=%s, connected=%s",
                        idx, port_label, port_id, idx in connected_indices)

            if idx in connected_indices:
                # Validate port structure
                if not port.get('id'):
                    logger.warning("Curate node %s: Port at index %d missing ID, skipping", node['id'], idx)
                    continue
                active_ports.append(port)
                port_index_map[port['id']] = idx
                logger.info("Curate node %s: Activated port %d -> '%s' (id=%s)",
                           node['id'], idx, port_label, port_id)

        if not active_ports:
            logger.warning("Curate node %s: No connected ports, skipping execution", node['id'])
            return None

        return {
            'model_type': model_type,
            'model_name': model_name,
            'all_ports': ports,  # All defined ports for template generation
            'active_ports': active_ports,  # Only connected ports for routing
            'port_index_map': port_index_map,
            'template': template,
            'parameters': parameters,
            'batch_size': parameters.get('batch_size', 1),
        }

    def _load_model(self, model_name: str, parameters: Dict, ctx: GraphExecutionContext):
        """Load VLM model with precision parameters."""
        try:
            precision_params = self._resolve_precision_params(model_name, parameters)
            model_adapter = ctx.get_model_func(model_name, precision_params)

            if not model_adapter or not model_adapter.is_loaded():
                logger.error("Model %s not available or not loaded", model_name)
                return None

            return model_adapter
        except Exception as exc:
            logger.exception("Failed to load model %s: %s", model_name, exc)
            return None

    def _gather_input_captions(self, node: Dict, ctx: GraphExecutionContext) -> Dict[str, str]:
        """Gather input captions from upstream nodes or use generated captions."""
        input_captions = {}

        for img_id in ctx.image_ids:
            caption = None

            # Check upstream connections (port 1 is caption input)
            for conn in ctx.plan.upstream.get(node['id'], []):
                if conn.to_port == 1:
                    buffered = ctx.buffers.get_output(conn.source_id, conn.from_port)
                    if buffered:
                        caption = str(buffered.get(img_id, '') or '')
                        if caption:
                            break

            # Fallback to generated captions
            if not caption:
                caption = ctx.captions.get(img_id, '')

            input_captions[img_id] = caption

        return input_captions

    def _build_prompt(self, config: Dict) -> str:
        """Build categorization prompt - uses ALL defined ports, not just connected ones."""
        template = config['template']
        all_ports = config['all_ports']  # Use all defined ports for prompt
        active_ports = config['active_ports']  # Connected ports

        # If custom template provided, use it with ALL ports
        if template and template.strip():
            prompt = self._apply_template(template, all_ports)
            logger.debug("Applied custom template with %d total ports (%d connected). Result: %s",
                        len(all_ports), len(active_ports), prompt[:200])
            return prompt

        # Build structured prompt using only ACTIVE ports for auto-generation
        categories_json = []
        for i, port in enumerate(active_ports):
            label = port.get('label', f'Port {i+1}')
            instruction = port.get('instruction', '').strip()
            categories_json.append({
                'label': label,
                'description': instruction if instruction else f'Category: {label}'
            })

        # Create prompt that asks for categorization
        categories_text = '\n'.join(
            f"{i+1}. {cat['label']}: {cat['description']}"
            for i, cat in enumerate(categories_json)
        )

        prompt = f"""Analyze this image and categorize it into ONE of the following categories:

{categories_text}

Respond with ONLY the category label (e.g., "{categories_json[0]['label']}").
Do not include explanations, numbers, or punctuation - just the exact label name."""

        logger.debug("Built auto-generated prompt with %d active categories", len(categories_json))
        return prompt

    async def _process_and_route(
        self,
        node: Dict,
        ctx: GraphExecutionContext,
        config: Dict,
        model_adapter,
        prompt: str,
        input_captions: Dict[str, str]
    ) -> Dict[str, Dict[str, str]]:
        """Process images in batches and route to appropriate ports."""
        all_ports = config['all_ports']  # All defined ports
        active_ports = config['active_ports']  # Only connected ports
        batch_size = config['batch_size']
        is_terminal = node['id'] in ctx.plan.terminal_nodes

        # Build set of active port IDs for quick lookup
        active_port_ids = {port['id'] for port in active_ports}

        # Initialize output buffers only for ACTIVE ports
        port_outputs = {port['id']: {} for port in active_ports}
        
        # Track routing stats for ALL ports (connected and disconnected)
        all_port_stats = {port['id']: 0 for port in all_ports}
        unmatched_count = 0
        processed_count = 0

        for batch_start in range(0, len(ctx.image_ids), batch_size):
            if ctx.should_cancel:
                logger.info("Execution cancelled at batch %d", batch_start)
                break

            batch_end = min(batch_start + batch_size, len(ctx.image_ids))
            batch_ids = ctx.image_ids[batch_start:batch_end]

            try:
                # Load images for this batch
                images, valid_indices = await self._load_batch_images(ctx, batch_ids, batch_start)

                if not images:
                    ctx.state.current_node_processed += len(batch_ids)
                    ctx.state.update()
                    continue

                # Process each image individually
                for i, img_idx in enumerate(valid_indices):
                    img_id = ctx.image_ids[img_idx]
                    image = images[i]
                    caption = input_captions.get(img_id, '')

                    # Replace {caption} placeholder with actual caption for this image
                    image_prompt = prompt.replace('{caption}', caption)

                    # Get VLM categorization
                    try:
                        vlm_response = model_adapter.generate_caption(
                            image, image_prompt, config['parameters']
                        )
                        vlm_response = str(vlm_response or '')

                        processed_count += 1
                        response_preview = (vlm_response or '').strip().replace('\n', ' ')
                        if len(response_preview) > 200:
                            response_preview = f"{response_preview[:197]}..."

                        logger.debug("Image %s VLM raw response: '%s'", img_id, vlm_response[:100])

                        # Route to appropriate port using ALL ports
                        matched_port = self._route_to_port(vlm_response, all_ports, img_id)
                        routed_label = "none"

                        if matched_port:
                            port_id = matched_port['id']
                            port_label = matched_port.get('label', 'Unknown')
                            
                            # Track stats for this port
                            all_port_stats[port_id] += 1

                            # Only save output if port is CONNECTED
                            if port_id in active_port_ids:
                                port_outputs[port_id][img_id] = caption

                                # Update context captions
                                ctx.captions[img_id] = caption

                                routed_label = port_label
                                logger.debug("Image %s routed to connected port '%s' (%s)",
                                             img_id, port_label, port_id)

                                # Persist immediately for live streaming
                                if is_terminal and caption:
                                    await self._persist_terminal_output(
                                        ctx, node, {img_id: caption},
                                        model_name=model_adapter.model_name,
                                        parameters=config['parameters'],
                                        extra_metadata={'model_type': 'vlm', 'port_id': port_id}
                                    )
                            else:
                                # Port is not connected
                                routed_label = f"{port_label} (disconnected)"
                                logger.debug("Image %s routed to disconnected port '%s' (%s)",
                                             img_id, port_label, port_id)
                        else:
                            unmatched_count += 1

                        #logger.info(
                        #    "Curate node %s image %s response='%s' routed_to='%s'",
                        #    node['id'],
                        #    img_id,
                        #    response_preview or '(empty)',
                        #    routed_label
                        #)

                    except Exception as exc:
                        logger.exception("Error processing image %s in curate node: %s", img_id, exc)

                    finally:
                        ctx.state.increment_progress()

                ctx.state.update()

            except Exception as exc:
                logger.exception("Batch error in curate node %s: %s", node['id'], exc)
                ctx.state.current_node_processed += len(batch_ids)
                ctx.state.update()

        # Log routing statistics
        self._log_routing_summary(
            node['id'],
            all_ports,
            active_port_ids,
            all_port_stats,
            processed_count,
            unmatched_count
        )

        return port_outputs

    def _route_to_port(self, vlm_response: str, ports: List[Dict], img_id: str) -> Optional[Dict]:
        """
        Route VLM response to appropriate port using explicit matching.

        Strategy:
        1. Exact label match (case-insensitive)
        2. Partial label match
        3. Keyword match from instruction
        """
        response_clean = vlm_response.strip().lower()

        # Remove common artifacts (numbers, colons, periods)
        response_clean = re.sub(r'^[\d\.\:\-\)\]\s]+', '', response_clean)
        response_clean = response_clean.strip()

        # Debug: Show available ports
        port_labels = [f"'{p.get('label', 'N/A')}'" for p in ports]
        logger.debug("Routing response '%s' against ports: %s", response_clean[:50], ', '.join(port_labels))

        # Strategy 1: Exact label match
        for port in ports:
            label = port.get('label', '').strip().lower()
            if label and response_clean == label:
                logger.debug("Image %s -> Port '%s' (exact match)", img_id, port.get('label'))
                return port

        # Strategy 2: Partial label match (label appears in response)
        for port in ports:
            label = port.get('label', '').strip().lower()
            if label and label in response_clean:
                logger.debug("Image %s -> Port '%s' (partial match)", img_id, port.get('label'))
                return port

        # Strategy 3: Check if response appears in label (reverse match)
        for port in ports:
            label = port.get('label', '').strip().lower()
            if response_clean and response_clean in label:
                logger.debug("Image %s -> Port '%s' (reverse match)", img_id, port.get('label'))
                return port

        # Strategy 4: Keyword matching from instruction
        for port in ports:
            instruction = port.get('instruction', '').strip().lower()
            if instruction:
                # Extract significant keywords (longer than 4 chars)
                keywords = [w for w in instruction.split() if len(w) > 4]
                if any(kw in response_clean for kw in keywords):
                    logger.debug("Image %s -> Port '%s' (keyword match)", img_id, port.get('label'))
                    return port

        if not ports:
            logger.error("Image %s: No ports available for routing", img_id)
        return None

    async def _load_batch_images(
        self,
        ctx: GraphExecutionContext,
        batch_ids: Sequence[str],
        batch_start: int
    ) -> Tuple[List, List[int]]:
        """Load images for a batch, returning valid images and their indices."""
        try:
            image_paths = await ctx.flow_hub.async_session.get_image_paths_batch(list(batch_ids))
            images = []
            valid_indices = []

            for i, img_id in enumerate(batch_ids):
                img_path = image_paths.get(img_id)
                if img_path:
                    try:
                        images.append(load_image(img_path))
                        valid_indices.append(batch_start + i)
                    except Exception as exc:
                        logger.error("Failed to load image %s: %s", img_id, exc)

            return images, valid_indices

        except Exception as exc:
            logger.exception("Error loading batch images: %s", exc)
            return [], []

    async def _finalize_outputs(
        self,
        node: Dict,
        ctx: GraphExecutionContext,
        config: Dict,
        port_outputs: Dict[str, Dict[str, str]]
    ) -> None:
        """Set outputs to buffers for downstream nodes."""
        # Set outputs for each ACTIVE port (for downstream nodes to consume)
        for port in config['active_ports']:
            port_id = port['id']
            port_index = config['port_index_map'].get(port_id)

            if port_index is not None:
                ctx.buffers.set_output(node['id'], port_index, port_outputs.get(port_id, {}))
        
        # Note: Persistence for live streaming is handled in _process_and_route
        
        # Note: Persistence for live streaming is handled in _process_and_route

    def _apply_template(self, template: str, ports: List[Dict]) -> str:
        """Apply port placeholder replacements for ALL defined ports."""
        logger.info("=== TEMPLATE APPLICATION START ===")
        logger.info("Input template: %s", template[:200])
        logger.info("Total ports: %d", len(ports))

        placeholders = {}

        # Create placeholders for ALL ports based on their array index
        for idx, port in enumerate(ports):
            port_id = port.get('id', f'port_{idx}')
            port_num = idx + 1  # 1-based port numbers
            raw_label = port.get('label', f'Port {port_num}')
            port_label = raw_label.strip() or f'Port {port_num}'
            port_instruction = (port.get('instruction') or '').strip()
            ref_key = port.get('refKey') or f"port_{port_num}"

            logger.debug("Port %d: id=%s, label='%s', instruction='%s'",
                         port_num, port_id, port_label,
                         port_instruction[:50] if port_instruction else '(empty)')

            # Combined reference text used for {port_refKey}
            if port_instruction:
                combined_text = f"{port_label}: {port_instruction}"
            else:
                combined_text = port_label

            # Add all possible placeholder formats
            placeholders[ref_key] = combined_text
            placeholders[f"port_{port_num}"] = combined_text

            # Legacy placeholders for backward compatibility
            placeholders[f"{ref_key}_label"] = port_label
            placeholders[f"port_{port_num}_label"] = port_label
            placeholders[f"{ref_key}_instruction"] = port_instruction
            placeholders[f"port_{port_num}_instruction"] = port_instruction

        logger.debug("Created placeholders: %s", list(placeholders.keys()))

        def replacer(match):
            key = match.group(1)
            replacement = placeholders.get(key, match.group(0))
            if replacement == match.group(0):
                logger.warning("Template placeholder {%s} not found", key)
            else:
                logger.debug("Replaced {%s} -> '%s'", key,
                           replacement[:30] if len(replacement) > 30 else replacement)
            return replacement

        result = re.sub(r'\{([^}]+)\}', replacer, template)
        logger.info("=== TEMPLATE RESULT ===")
        logger.info("Output: %s", result[:300])
        logger.info("=== END TEMPLATE APPLICATION ===")
        return result

    def _log_routing_summary(
        self,
        node_id: str,
        all_ports: List[Dict],
        active_port_ids: Set[str],
        all_port_stats: Dict[str, int],
        processed_count: int,
        unmatched_count: int
    ) -> None:
        """Log routing statistics for observability."""
        connected_parts = []
        disconnected_parts = []

        for port in all_ports:
            port_id = port['id']
            count = all_port_stats.get(port_id, 0)
            label = port.get('label', port_id)
            
            if count > 0:
                if port_id in active_port_ids:
                    connected_parts.append(f"{label}: {count}")
                else:
                    disconnected_parts.append(f"{label}: {count}")

        # Build summary message
        connected_summary = ", ".join(connected_parts) if connected_parts else "none"
        
        log_msg = f"{processed_count} images curated to {connected_summary}"
        
        if disconnected_parts:
            disconnected_summary = ", ".join(disconnected_parts)
            log_msg += f" | disconnected ports: {disconnected_summary}"
        
        if unmatched_count > 0:
            log_msg += f" | couldn't match: {unmatched_count}"

        logger.info(log_msg)

    @staticmethod
    def _resolve_precision_params(model_name: str, parameters: Dict) -> Optional[Dict]:
        """Resolve precision parameters from node config and model defaults."""
        if 'precision' not in parameters and 'use_flash_attention' not in parameters:
            return None

        from app.models import get_factory

        factory = get_factory()
        defaults = factory.get_precision_defaults(model_name) or {}

        return {
            'precision': parameters.get('precision', defaults.get('precision')),
            'use_flash_attention': parameters.get('use_flash_attention', defaults.get('use_flash_attention'))
        }


class AimodelNodeExecutor(BaseNodeExecutor):
    node_type = 'aimodel'

    async def run(self, node: Dict, ctx: GraphExecutionContext) -> None:
        model_name = node['data'].get('model', 'blip')
        parameters = node['data'].get('parameters', {}).copy()
        batch_size = parameters.get('batch_size', 1)

        base_prompt = ctx.prompt_resolver.get_static_prompt(node)
        precision_params = self._resolve_precision_params(model_name, parameters)

        model_adapter = ctx.get_model_func(model_name, precision_params)
        if not model_adapter or not model_adapter.is_loaded():
            raise ValueError(f"Model {model_name} not available")

        prev_captions = [ctx.captions.get(img_id, '') for img_id in ctx.image_ids]
        is_terminal = node['id'] in ctx.plan.terminal_nodes

        ctx.state.set_current_node(node['id'])
        ctx.state.update()

        for batch_start in range(0, len(ctx.image_ids), batch_size):
            if ctx.should_cancel:
                logger.info("Execution cancelled before processing batch starting at %s", batch_start)
                break

            batch_end = min(batch_start + batch_size, len(ctx.image_ids))
            batch_ids = ctx.image_ids[batch_start:batch_end]

            try:
                images, valid_indices = await self._load_batch_images(ctx, batch_ids, batch_start)
                if not images:
                    self._mark_failed_batch(ctx, len(batch_ids), is_terminal)
                    continue

                prompts = self._build_batch_prompts(
                    node, prev_captions, valid_indices, base_prompt, ctx.prompt_resolver
                )

                new_captions = await self._generate_batch_captions(
                    images, prompts, model_adapter, parameters, batch_size
                )

                self._update_captions_and_state(ctx, valid_indices, new_captions)

                text_outputs = {
                    ctx.image_ids[idx]: caption
                    for idx, caption in zip(valid_indices, new_captions)
                }
                ctx.record_text_output(node['id'], text_outputs)

                # Persist immediately for live streaming when connected to Output
                if is_terminal and text_outputs:
                    await self._persist_terminal_output(
                        ctx, node, text_outputs,
                        model_name=model_adapter.model_name,
                        parameters=parameters
                    )

                if batch_start % batch_size == 0:
                    ctx.state.update()

            except Exception as exc:
                logger.exception("Batch error in node %s: %s", node['id'], exc)
                self._mark_failed_batch(ctx, len(batch_ids), is_terminal)

        ctx.state.update()

    async def _load_batch_images(
        self,
        ctx: GraphExecutionContext,
        batch_ids: Sequence[str],
        batch_start: int
    ) -> Tuple[List, List[int]]:
        image_paths = await ctx.flow_hub.async_session.get_image_paths_batch(list(batch_ids))
        images = []
        valid_indices = []
        for i, img_id in enumerate(batch_ids):
            img_path = image_paths.get(img_id)
            if img_path:
                images.append(load_image(img_path))
                valid_indices.append(batch_start + i)
        return images, valid_indices

    def _build_batch_prompts(
        self,
        node: Dict,
        prev_captions: List[str],
        valid_indices: List[int],
        base_prompt: str,
        prompt_resolver
    ) -> List[Optional[str]]:
        return [
            prompt_resolver.get_prompt_for_image(node, prev_captions, idx, base_prompt)
            for idx in valid_indices
        ]

    async def _generate_batch_captions(
        self,
        images: List,
        prompts: List[Optional[str]],
        model_adapter,
        parameters: Dict,
        batch_size: int
    ) -> List[str]:
        if len(images) > 1 and batch_size > 1:
            return model_adapter.generate_captions_batch(images, prompts, parameters)
        return [model_adapter.generate_caption(images[0], prompts[0], parameters)]

    def _update_captions_and_state(
        self,
        ctx: GraphExecutionContext,
        valid_indices: List[int],
        new_captions: List[str]
    ) -> None:
        for idx, caption in zip(valid_indices, new_captions):
            ctx.captions[ctx.image_ids[idx]] = caption
            ctx.state.increment_progress()

    def _mark_failed_batch(self, ctx: GraphExecutionContext, batch_size: int, is_terminal: bool) -> None:
        for _ in range(batch_size):
            ctx.state.increment_progress()
            if is_terminal:
                ctx.state.increment(success=False)

    @staticmethod
    def _resolve_precision_params(model_name: str, parameters: Dict) -> Optional[Dict]:
        if 'precision' not in parameters and 'use_flash_attention' not in parameters:
            return None

        from app.models import get_factory

        factory = get_factory()
        defaults = factory.get_precision_defaults(model_name) or {}

        return {
            'precision': parameters.get('precision', defaults.get('precision')),
            'use_flash_attention': parameters.get('use_flash_attention', defaults.get('use_flash_attention'))
        }


class OutputNodeExecutor(BaseNodeExecutor):
    """
    Terminal node that marks the end of graph execution.
    
    When nodes are connected to Output, they become "terminal nodes" and persist
    their results immediately for live streaming. This Output node serves as:
    1. A visual endpoint in the graph
    2. A fallback for persistence if no terminal nodes exist
    """
    node_type = 'output'

    async def run(self, node: Dict, ctx: GraphExecutionContext) -> None:
        ctx.state.set_current_node(node['id'])
        
        # If terminal nodes exist, they already persisted results for live streaming
        if ctx.plan.terminal_nodes:
            logger.info(
                "Output node %s: %d terminal nodes already persisted results",
                node['id'],
                len(ctx.plan.terminal_nodes)
            )
            ctx.state.update()
            return

        # Fallback: persist from buffers if no terminal nodes (edge case)
        logger.info("Output node %s: no terminal nodes, using fallback persistence", node['id'])
        await self._fallback_persist(ctx, node)
        ctx.state.update()

    async def _fallback_persist(self, ctx: GraphExecutionContext, node: Dict) -> None:
        """Fallback persistence when no terminal nodes exist."""
        upstream_conns = ctx.plan.upstream.get(node['id'], [])
        
        if not upstream_conns:
            logger.warning("Output node %s has no upstream connections", node['id'])
            return
        
        # Collect from all upstream
        collected_outputs: Dict[str, str] = {}
        for conn in upstream_conns:
            buffered = ctx.buffers.get_output(conn.source_id, conn.from_port)
            if buffered:
                for img_id, value in buffered.items():
                    if value:
                        collected_outputs[img_id] = str(value)
        
        if collected_outputs:
            await self._persist_terminal_output(
                ctx, node, collected_outputs,
                model_name="graph_output"
            )


class NodeExecutorRegistry:
    """
    Lightweight registry so GraphExecutor can stay generic.
    """

    def __init__(self):
        self._executors = {
            cls.node_type: cls()
            for cls in (
                InputNodeExecutor,
                PromptNodeExecutor,
                ConjunctionNodeExecutor,
                CurateNodeExecutor,
                AimodelNodeExecutor,
                OutputNodeExecutor,
            )
        }

    def get(self, node_type: str) -> BaseNodeExecutor:
        if node_type not in self._executors:
            raise ValueError(f"No executor registered for node type: {node_type}")
        return self._executors[node_type]
