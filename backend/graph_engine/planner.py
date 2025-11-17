"""
Build execution plans for validated graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

from .schema import DataType, ValidatedGraph


@dataclass(frozen=True)
class PlannedConnection:
    source_id: str
    target_id: str
    from_port: int
    to_port: int
    data_type: DataType


@dataclass
class ExecutionPlan:
    ordered_nodes: List[str]
    upstream: Dict[str, List[PlannedConnection]]
    downstream: Dict[str, List[PlannedConnection]]
    terminal_nodes: Set[str]


class PlanBuilder:
    """Turns a validated graph into an acyclic execution plan."""

    def __init__(self, schemas=None):
        self.schemas = schemas

    def build(self, graph: ValidatedGraph) -> ExecutionPlan:
        nodes = graph.nodes_by_id
        schema_lookup = self.schemas
        connections: List[PlannedConnection] = []
        for raw in graph.connections:
            source_id = raw['from']
            target_id = raw['to']
            from_port = raw.get('fromPort', 0)
            to_port = raw.get('toPort', 0)

            source_schema = None
            target_schema = None
            if schema_lookup:
                source_schema = schema_lookup.get(nodes[source_id]['type'])
            if schema_lookup:
                target_schema = schema_lookup.get(nodes[target_id]['type'])

            data_type = DataType.ANY
            if source_schema and from_port in source_schema.outputs:
                data_type = source_schema.outputs[from_port].data_type

            connections.append(
                PlannedConnection(
                    source_id=source_id,
                    target_id=target_id,
                    from_port=from_port,
                    to_port=to_port,
                    data_type=data_type,
                )
            )

        downstream: Dict[str, List[PlannedConnection]] = {node_id: [] for node_id in nodes}
        upstream: Dict[str, List[PlannedConnection]] = {node_id: [] for node_id in nodes}
        for conn in connections:
            downstream[conn.source_id].append(conn)
            upstream[conn.target_id].append(conn)

        nodes_on_path = self._prune_to_paths(graph.input_node_id, graph.output_node_id, downstream, upstream)
        ordered_nodes = self._topological_order(nodes_on_path, downstream, upstream)
        terminal_nodes = {
            conn.source_id
            for conn in upstream[graph.output_node_id]
            if conn.source_id in nodes_on_path
        }

        # Filter upstream/downstream maps to relevant nodes only
        filtered_upstream = {node_id: upstream[node_id] for node_id in nodes_on_path}
        filtered_downstream = {node_id: downstream[node_id] for node_id in nodes_on_path}

        return ExecutionPlan(
            ordered_nodes=ordered_nodes,
            upstream=filtered_upstream,
            downstream=filtered_downstream,
            terminal_nodes=terminal_nodes,
        )

    @staticmethod
    def _prune_to_paths(
        input_id: str,
        output_id: str,
        downstream: Dict[str, List[PlannedConnection]],
        upstream: Dict[str, List[PlannedConnection]],
    ) -> Set[str]:
        reachable_from_input: Set[str] = set()
        queue = [input_id]
        while queue:
            node_id = queue.pop()
            if node_id in reachable_from_input:
                continue
            reachable_from_input.add(node_id)
            for conn in downstream.get(node_id, []):
                queue.append(conn.target_id)

        reachable_to_output: Set[str] = set()
        queue = [output_id]
        while queue:
            node_id = queue.pop()
            if node_id in reachable_to_output:
                continue
            reachable_to_output.add(node_id)
            for conn in upstream.get(node_id, []):
                queue.append(conn.source_id)

        intersection = reachable_from_input & reachable_to_output
        if output_id not in intersection:
            raise ValueError("Output node is not reachable from Input node")
        return intersection

    @staticmethod
    def _topological_order(
        nodes_on_path: Set[str],
        downstream: Dict[str, List[PlannedConnection]],
        upstream: Dict[str, List[PlannedConnection]],
    ) -> List[str]:
        indegree = {node_id: 0 for node_id in nodes_on_path}
        for node_id in nodes_on_path:
            indegree[node_id] = sum(
                1 for conn in upstream.get(node_id, []) if conn.source_id in nodes_on_path
            )

        queue = [node_id for node_id, degree in indegree.items() if degree == 0]
        ordered = []

        while queue:
            node_id = queue.pop(0)
            ordered.append(node_id)
            for conn in downstream.get(node_id, []):
                if conn.target_id not in nodes_on_path:
                    continue
                indegree[conn.target_id] -= 1
                if indegree[conn.target_id] == 0:
                    queue.append(conn.target_id)

        if len(ordered) != len(nodes_on_path):
            raise ValueError("Graph contains cycles or disconnected components")
        return ordered
