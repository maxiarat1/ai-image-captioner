"""
Registered node executors for the graph runtime.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Sequence, Tuple

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
    node_type: str

    async def run(self, node: Dict, ctx: GraphExecutionContext) -> None:
        raise NotImplementedError


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

        if node['id'] in ctx.plan.terminal_nodes:
            for _ in ctx.image_ids:
                ctx.state.increment(success=bool(text))
            await self._persist_results(ctx, node, outputs)
            ctx.state.update()

    async def _persist_results(
        self,
        ctx: GraphExecutionContext,
        node: Dict,
        outputs: Dict[str, str]
    ) -> None:
        processed_batch = []
        for sequence_num, img_id in enumerate(ctx.image_ids):
            caption = outputs.get(img_id, '')
            if not caption:
                continue
            processed_batch.append(
                create_processed_data(
                    image_id=img_id,
                    content=caption,
                    model_name="prompt",
                    parameters={},
                    metadata={'node_id': node['id'], 'node_type': 'prompt'},
                    sequence_num=sequence_num,
                    data_type='text'
                )
            )

        if processed_batch:
            await ctx.flow_hub.route_batch(processed_batch)


class ConjunctionNodeExecutor(BaseNodeExecutor):
    node_type = 'conjunction'

    async def run(self, node: Dict, ctx: GraphExecutionContext) -> None:
        template = node['data'].get('template', '')
        connected_items = node['data'].get('connectedItems') or []

        ctx.state.set_current_node(node['id'])
        ctx.state.current_node_processed = len(ctx.image_ids)
        ctx.state.update()

        reference_values = self._build_reference_values(node, connected_items, ctx)
        resolved_outputs = self._resolve_outputs(node, template, reference_values, ctx)

        if not resolved_outputs:
            logger.warning("Conjunction node %s produced no outputs", node['id'])
            return

        ctx.buffers.set_output(node['id'], NodePort.DEFAULT_OUTPUT, resolved_outputs)

        is_final_node = node['id'] in ctx.plan.terminal_nodes
        if is_final_node:
            for img_id in ctx.image_ids:
                ctx.state.increment(success=bool(resolved_outputs.get(img_id)))
            await self._persist_results(ctx, node, resolved_outputs)

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

    async def _persist_results(
        self,
        ctx: GraphExecutionContext,
        node: Dict,
        outputs: Dict[str, str]
    ) -> None:
        processed_batch = []

        for sequence_num, img_id in enumerate(ctx.image_ids):
            caption = outputs.get(img_id, '')
            if not caption:
                continue

            processed_batch.append(
                create_processed_data(
                    image_id=img_id,
                    content=caption,
                    model_name="conjunction",
                    parameters={},
                    metadata={
                        'node_id': node['id'],
                        'node_type': 'conjunction',
                        'template': node['data'].get('template', '')
                    },
                    sequence_num=sequence_num,
                    data_type='text'
                )
            )

        if processed_batch:
            await ctx.flow_hub.route_batch(processed_batch)


class CurateNodeExecutor(BaseNodeExecutor):
    node_type = 'curate'

    async def run(self, node: Dict, ctx: GraphExecutionContext) -> None:
        model_type = node['data'].get('modelType', 'vlm')
        model_name = node['data'].get('model')
        ports = node['data'].get('ports', [])
        template = node['data'].get('template', '')
        parameters = node['data'].get('parameters', {}).copy()
        batch_size = parameters.get('batch_size', 1)

        if not model_name:
            logger.error("Curate node %s has no model selected", node['id'])
            return

        if not ports:
            logger.error("Curate node %s has no routing ports defined", node['id'])
            return

        # Currently only VLM is supported
        if model_type != 'vlm':
            logger.error("Curate node %s: Only VLM mode is currently supported", node['id'])
            return

        ctx.state.set_current_node(node['id'])
        ctx.state.update()

        # Load the VLM model
        precision_params = self._resolve_precision_params(model_name, parameters)
        model_adapter = ctx.get_model_func(model_name, precision_params)
        if not model_adapter or not model_adapter.is_loaded():
            raise ValueError(f"Model {model_name} not available")

        # Get captions from upstream nodes
        input_captions = {}
        for img_id in ctx.image_ids:
            caption = None
            for conn in ctx.plan.upstream.get(node['id'], []):
                if conn.to_port == 1:  # Caption input port
                    buffered = ctx.buffers.get_output(conn.source_id, conn.from_port)
                    if buffered:
                        caption = str(buffered.get(img_id, '') or '')
                        if caption:
                            break
            
            # Fallback to generated captions
            if not caption:
                caption = ctx.captions.get(img_id, '')
            
            input_captions[img_id] = caption

        # Build categorization prompt
        categorization_prompt = self._build_categorization_prompt(ports)

        # Process images in batches
        port_outputs = {port['id']: {} for port in ports}
        
        for batch_start in range(0, len(ctx.image_ids), batch_size):
            if ctx.should_cancel:
                logger.info("Execution cancelled before processing batch starting at %s", batch_start)
                break

            batch_end = min(batch_start + batch_size, len(ctx.image_ids))
            batch_ids = ctx.image_ids[batch_start:batch_end]

            try:
                # Load images
                images, valid_indices = await self._load_batch_images(ctx, batch_ids, batch_start)
                if not images:
                    ctx.state.current_node_processed += len(batch_ids)
                    continue

                # Evaluate each image
                for i, img_idx in enumerate(valid_indices):
                    img_id = ctx.image_ids[img_idx]
                    image = images[i]
                    input_caption = input_captions.get(img_id, '')

                    # Ask VLM to categorize the image
                    vlm_response = model_adapter.generate_caption(image, categorization_prompt, parameters)
                    
                    # Determine which port(s) to route to based on VLM response
                    matched_ports = self._match_response_to_ports(vlm_response, ports)
                    
                    # Route to matched ports
                    for port in matched_ports:
                        port_id = port['id']
                        
                        # Build output using template or pass through caption
                        if template:
                            output = self._apply_template(template, port, input_caption, ports)
                        else:
                            output = input_caption
                        
                        port_outputs[port_id][img_id] = output
                        
                        # Update context captions with the output
                        ctx.captions[img_id] = output

                    ctx.state.increment_progress()

                ctx.state.update()

            except Exception as exc:
                logger.exception("Batch error in curate node %s: %s", node['id'], exc)
                ctx.state.current_node_processed += len(batch_ids)

        # Set outputs for each port
        for port in ports:
            port_id = port['id']
            port_num = int(port_id) if isinstance(port_id, str) and port_id.isdigit() else port_id
            ctx.buffers.set_output(node['id'], port_num, port_outputs[port_id])

        # If terminal node, persist results from all ports
        if node['id'] in ctx.plan.terminal_nodes:
            await self._persist_all_port_results(ctx, node, port_outputs, model_adapter, parameters)

        ctx.state.update()

    def _build_categorization_prompt(self, ports: List[Dict]) -> str:
        """Build a prompt that asks the VLM to categorize the image into one of the ports."""
        categories = []
        for i, port in enumerate(ports):
            label = port.get('label', f'Port {i+1}')
            instruction = port.get('instruction', '').strip()
            if instruction:
                categories.append(f"{label}: {instruction}")
            else:
                categories.append(label)
        
        categories_text = '\n'.join(f"- {cat}" for cat in categories)
        
        prompt = f"""Analyze this image and determine which category best describes it.

Available categories:
{categories_text}

Respond with ONLY the category name (e.g., "{ports[0].get('label', 'Port 1')}"). Do not add explanations."""
        
        return prompt

    def _match_response_to_ports(self, vlm_response: str, ports: List[Dict]) -> List[Dict]:
        """Match VLM response to port(s) based on label or instruction keywords."""
        matched = []
        response_lower = vlm_response.lower().strip()
        
        # Try to match by port label first
        for port in ports:
            label = port.get('label', '').lower().strip()
            if label and label in response_lower:
                matched.append(port)
                return matched  # Return first match
        
        # If no label match, try matching by instruction keywords
        for port in ports:
            instruction = port.get('instruction', '').lower().strip()
            if instruction:
                # Extract key words from instruction
                keywords = [w for w in instruction.split() if len(w) > 3]
                if any(keyword in response_lower for keyword in keywords):
                    matched.append(port)
                    return matched
        
        # If still no match, default to first port
        if not matched and ports:
            logger.warning("Could not match VLM response '%s' to any port, using first port", vlm_response[:50])
            matched.append(ports[0])
        
        return matched

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

    def _apply_template(
        self,
        template: str,
        current_port: Dict,
        caption: str,
        all_ports: List[Dict]
    ) -> str:
        """Apply template with port-specific placeholders."""
        result = template
        
        # Build placeholder values
        placeholders = {
            'caption': caption,
            current_port['refKey']: current_port.get('label', ''),
            f"{current_port['refKey']}_label": current_port.get('label', ''),
            f"{current_port['refKey']}_instruction": current_port.get('instruction', ''),
        }
        
        # Replace placeholders
        import re
        placeholder_pattern = re.compile(r'\{([^}]+)\}')
        
        def replacer(match):
            key = match.group(1)
            return placeholders.get(key, match.group(0))
        
        return placeholder_pattern.sub(replacer, result)

    async def _persist_all_port_results(
        self,
        ctx: GraphExecutionContext,
        node: Dict,
        port_outputs: Dict[str, Dict[str, str]],
        model_adapter,
        parameters: Dict
    ) -> None:
        """Persist results from all ports to database."""
        processed_batch = []
        
        for port_id, outputs in port_outputs.items():
            for img_id, caption in outputs.items():
                if not caption:
                    continue
                
                processed_batch.append(
                    create_processed_data(
                        image_id=img_id,
                        content=caption,
                        model_name=model_adapter.model_name,
                        parameters=parameters,
                        metadata={
                            'node_id': node['id'],
                            'node_type': 'curate',
                            'port_id': port_id,
                            'model_type': node['data'].get('modelType', 'vlm')
                        },
                        sequence_num=ctx.image_ids.index(img_id),
                        data_type='text'
                    )
                )
        
        if processed_batch:
            await ctx.flow_hub.route_batch(processed_batch)
            for _ in processed_batch:
                ctx.state.increment(success=True)

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
        is_final_node = node['id'] in ctx.plan.terminal_nodes

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
                    self._mark_failed_batch(ctx, len(batch_ids), is_final_node)
                    continue

                prompts = self._build_batch_prompts(
                    node, prev_captions, valid_indices, base_prompt, ctx.prompt_resolver
                )

                new_captions = await self._generate_batch_captions(
                    images, prompts, model_adapter, parameters, batch_size
                )

                self._update_captions_and_state(
                    ctx, valid_indices, new_captions, is_final_node
                )

                text_outputs = {
                    ctx.image_ids[idx]: caption
                    for idx, caption in zip(valid_indices, new_captions)
                }
                ctx.record_text_output(node['id'], text_outputs)

                if is_final_node and text_outputs:
                    await self._persist_results(ctx, text_outputs, model_adapter, node, parameters)

                if batch_start % batch_size == 0:
                    ctx.state.update()

            except Exception as exc:
                logger.exception("Batch error in node %s: %s", node['id'], exc)
                self._mark_failed_batch(ctx, len(batch_ids), is_final_node)

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
        new_captions: List[str],
        is_final_node: bool
    ) -> None:
        for idx, caption in zip(valid_indices, new_captions):
            ctx.captions[ctx.image_ids[idx]] = caption
            ctx.state.increment_progress()
            if is_final_node:
                ctx.state.increment(success=True)

    def _mark_failed_batch(self, ctx: GraphExecutionContext, batch_size: int, is_final_node: bool) -> None:
        for _ in range(batch_size):
            ctx.state.increment_progress()
            if is_final_node:
                ctx.state.increment(success=False)

    async def _persist_results(
        self,
        ctx: GraphExecutionContext,
        outputs: Dict[str, str],
        model_adapter,
        node: Dict,
        parameters: Dict
    ) -> None:
        processed_batch = [
            create_processed_data(
                image_id=image_id,
                content=caption,
                model_name=model_adapter.model_name,
                parameters=parameters,
                metadata={'node_id': node['id'], 'node_type': 'aimodel'},
                sequence_num=ctx.image_ids.index(image_id),
                data_type='text'
            )
            for image_id, caption in outputs.items()
        ]
        await ctx.flow_hub.route_batch(processed_batch)

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
    node_type = 'output'

    async def run(self, node: Dict, ctx: GraphExecutionContext) -> None:
        if ctx.plan.terminal_nodes:
            logger.info(
                "Skipping Output node save; %d processing nodes already persisted results",
                len(ctx.plan.terminal_nodes)
            )
            return

        logger.info("Output node saving captions as fallback path")
        processed_batch = []

        for sequence_num, img_id in enumerate(ctx.image_ids):
            caption = ctx.captions.get(img_id, '')
            if not caption:
                continue
            processed_batch.append(
                create_processed_data(
                    image_id=img_id,
                    content=caption,
                    model_name="graph_output",
                    parameters={},
                    metadata={'output_node_id': node['id']},
                    sequence_num=sequence_num,
                    data_type='text'
                )
            )

        if processed_batch:
            routed = await ctx.flow_hub.route_batch(processed_batch)
            logger.info("Routed %s/%s captions through Flow Control Hub", routed, len(processed_batch))


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
