"""
Graph schema definitions and validation helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class DataType(str, Enum):
    IMAGE = "image"
    TEXT = "text"
    METADATA = "metadata"
    ANY = "any"


@dataclass(frozen=True)
class PortSchema:
    port: int
    data_type: DataType
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class NodeSchema:
    type: str
    inputs: Dict[int, PortSchema]
    outputs: Dict[int, PortSchema]
    allow_additional_inputs: bool = False
    allow_additional_outputs: bool = False


NODE_SCHEMAS: Dict[str, NodeSchema] = {
    'input': NodeSchema(
        type='input',
        inputs={},
        outputs={
            0: PortSchema(0, DataType.IMAGE, description="Images to process"),
        }
    ),
    'aimodel': NodeSchema(
        type='aimodel',
        inputs={
            0: PortSchema(0, DataType.IMAGE, description="Images to analyse"),
            1: PortSchema(1, DataType.TEXT, required=False, description="Prompt override"),
        },
        outputs={
            0: PortSchema(0, DataType.TEXT, description="Generated captions"),
        }
    ),
    'prompt': NodeSchema(
        type='prompt',
        inputs={},
        outputs={
            0: PortSchema(0, DataType.TEXT, description="Static prompt text"),
        }
    ),
    'conjunction': NodeSchema(
        type='conjunction',
        inputs={
            0: PortSchema(0, DataType.TEXT, required=False, description="Primary text input"),
            1: PortSchema(1, DataType.TEXT, required=False, description="Secondary text input"),
        },
        outputs={
            0: PortSchema(0, DataType.TEXT, description="Templated text result"),
        },
        allow_additional_inputs=True,
    ),
    'curate': NodeSchema(
        type='curate',
        inputs={
            0: PortSchema(0, DataType.IMAGE, description="Images to route"),
            1: PortSchema(1, DataType.TEXT, required=False, description="Optional caption input"),
        },
        outputs={
            0: PortSchema(0, DataType.TEXT, description="Curated captions"),
        },
        allow_additional_outputs=True
    ),
    'output': NodeSchema(
        type='output',
        inputs={
            0: PortSchema(0, DataType.TEXT, description="Final captions"),
        },
        outputs={}
    ),
}


class GraphValidationError(ValueError):
    """Raised when a graph definition fails validation."""


@dataclass
class ValidatedGraph:
    nodes_by_id: Dict[str, Dict]
    connections: List[Dict]
    input_node_id: str
    output_node_id: str


class GraphValidator:
    """
    Validates raw graph payloads emitted by the frontend canvas.
    """

    def __init__(self, schemas: Optional[Dict[str, NodeSchema]] = None):
        self.schemas = schemas or NODE_SCHEMAS

    def validate(self, graph: Dict) -> ValidatedGraph:
        nodes = graph.get('nodes') or []
        conns = graph.get('connections') or []
        if not nodes:
            raise GraphValidationError("Graph contains no nodes")

        nodes_by_id = {node['id']: node for node in nodes}

        input_nodes = [n for n in nodes if n.get('type') == 'input']
        output_nodes = [n for n in nodes if n.get('type') == 'output']
        if not input_nodes or not output_nodes:
            raise GraphValidationError("Graph must contain Input and Output nodes")

        # Use the first occurrences (UI enforces singletons)
        input_node_id = input_nodes[0]['id']
        output_node_id = output_nodes[0]['id']

        for conn in conns:
            source_id = conn.get('from')
            target_id = conn.get('to')
            from_port = conn.get('fromPort', 0)
            to_port = conn.get('toPort', 0)

            if source_id not in nodes_by_id or target_id not in nodes_by_id:
                raise GraphValidationError(f"Connection references unknown node: {conn}")

            source_node = nodes_by_id[source_id]
            target_node = nodes_by_id[target_id]
            source_schema = self._get_schema(source_node['type'])
            target_schema = self._get_schema(target_node['type'])

            if from_port not in source_schema.outputs:
                if not source_schema.allow_additional_outputs:
                    raise GraphValidationError(
                        f"Node {source_node['type']} does not expose output port {from_port}"
                    )

            if to_port not in target_schema.inputs:
                if not target_schema.allow_additional_inputs:
                    raise GraphValidationError(
                        f"Node {target_node['type']} does not expose input port {to_port}"
                    )
                continue  # Skip strict type checking for dynamic inputs

            from_type = source_schema.outputs[from_port].data_type
            to_type = target_schema.inputs[to_port].data_type
            if not self._is_compatible(from_type, to_type):
                raise GraphValidationError(
                    f"Incompatible connection {source_node['type']}:{from_port} -> "
                    f"{target_node['type']}:{to_port} ({from_type} -> {to_type})"
                )

        return ValidatedGraph(
            nodes_by_id=nodes_by_id,
            connections=conns,
            input_node_id=input_node_id,
            output_node_id=output_node_id
        )

    def _get_schema(self, node_type: str) -> NodeSchema:
        if node_type not in self.schemas:
            raise GraphValidationError(f"Unsupported node type: {node_type}")
        return self.schemas[node_type]

    @staticmethod
    def _is_compatible(source: DataType, target: DataType) -> bool:
        if DataType.ANY in (source, target):
            return True
        return source == target
