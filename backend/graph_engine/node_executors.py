"""
Registered node executors for the graph runtime.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

from flow_control import create_processed_data
from utils.image_utils import load_image

from .context import GraphExecutionContext
from .constants import NodePort

logger = logging.getLogger(__name__)


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
        ctx.buffers.set_output(
            node['id'],
            NodePort.DEFAULT_OUTPUT,
            {img_id: node['data'].get('text', '') for img_id in ctx.image_ids}
        )


class ConjunctionNodeExecutor(BaseNodeExecutor):
    node_type = 'conjunction'

    async def run(self, node: Dict, ctx: GraphExecutionContext) -> None:
        # Conjunction resolution happens lazily inside PromptResolver.
        logger.debug("Conjunction node %s evaluated lazily via PromptResolver", node['id'])


class CurateNodeExecutor(BaseNodeExecutor):
    node_type = 'curate'

    async def run(self, node: Dict, ctx: GraphExecutionContext) -> None:
        logger.warning(
            "Curate node %s is not yet implemented. Images flow through default path.",
            node['id']
        )


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
