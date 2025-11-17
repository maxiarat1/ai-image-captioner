"""
Graph Executor - orchestrates node execution using the declarative runtime.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from database import ExecutionManager
from flow_control import FlowControlHub
from graph_engine import GraphValidator, PlanBuilder
from graph_engine.context import GraphExecutionContext
from graph_engine.node_executors import NodeExecutorRegistry
from graph_engine.prompting import PromptResolver
from graph_engine.schema import GraphValidationError, NODE_SCHEMAS
from graph_engine.state import ExecutionState

logger = logging.getLogger(__name__)


class GraphExecutor:
    """
    High-level coordinator that wires validation, planning, and node execution.
    """

    def __init__(self, exec_manager: ExecutionManager, flow_hub: FlowControlHub):
        self.exec_manager = exec_manager
        self.flow_hub = flow_hub
        self.validator = GraphValidator(NODE_SCHEMAS)
        self.plan_builder = PlanBuilder(NODE_SCHEMAS)
        self.registry = NodeExecutorRegistry()
        self.should_cancel = False
        self._active_context: Optional[GraphExecutionContext] = None

    async def execute(self, job_id: str, get_model_func: Callable) -> None:
        job = self.exec_manager.get_job(job_id)
        if not job:
            logger.error("Job %s not found", job_id)
            return

        try:
            self.exec_manager.update_status(job_id, status='running')
            image_ids = job['image_ids']
            graph = job['graph']

            validated = self.validator.validate(graph)
            plan = self.plan_builder.build(validated)
            prompt_resolver = PromptResolver(list(validated.nodes_by_id.values()), validated.connections)
            state = ExecutionState(self.exec_manager, job_id, len(image_ids))

            ctx = GraphExecutionContext(
                image_ids=image_ids,
                plan=plan,
                state=state,
                flow_hub=self.flow_hub,
                get_model_func=get_model_func,
                prompt_resolver=prompt_resolver,
            )
            ctx.should_cancel = self.should_cancel
            self._active_context = ctx

            for node_id in plan.ordered_nodes:
                if self.should_cancel:
                    ctx.should_cancel = True
                    break

                node = validated.nodes_by_id[node_id]
                executor = self.registry.get(node['type'])
                logger.info("Executing node: %s (%s)", node['type'], node_id)
                await executor.run(node, ctx)

            if self.should_cancel:
                logger.info("Job %s cancelled", job_id)
                self.exec_manager.update_status(job_id, status='cancelled')
            else:
                logger.info("Job %s completed", job_id)
                self.exec_manager.update_status(
                    job_id,
                    status='completed',
                    processed=len(image_ids),
                    success=state.success,
                    failed=state.failed
                )

        except GraphValidationError as exc:
            logger.error("Graph validation failed: %s", exc)
            self.exec_manager.update_status(job_id, status='failed', error=str(exc))
        except Exception as exc:
            logger.exception("Graph execution failed: %s", exc)
            self.exec_manager.update_status(job_id, status='failed', error=str(exc))
            raise
        finally:
            self._active_context = None

    def cancel(self) -> None:
        self.should_cancel = True
        if self._active_context:
            self._active_context.should_cancel = True
