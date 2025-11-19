"""
Shared execution context passed to node executors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from flow_control import FlowControlHub
from .data_store import GraphDataStore
from .planner import ExecutionPlan
from .state import ExecutionState


ModelLoader = Callable[[str, Optional[Dict]], object]


@dataclass
class GraphExecutionContext:
    image_ids: List[str]
    plan: ExecutionPlan
    state: ExecutionState
    flow_hub: FlowControlHub
    get_model_func: ModelLoader
    prompt_resolver: "PromptResolver"
    nodes_by_id: Dict[str, Dict]

    def __post_init__(self):
        self.buffers = GraphDataStore(self.image_ids)
        self.captions: Dict[str, str] = {img_id: '' for img_id in self.image_ids}
        self.errors: List[str] = []
        self.should_cancel = False

    def record_text_output(self, node_id: str, values: Dict[str, str]) -> None:
        self.buffers.set_output(node_id, 0, values)
        self.captions.update(values)
