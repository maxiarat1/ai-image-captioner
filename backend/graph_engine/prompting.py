"""
Prompt resolution utilities extracted from the legacy executor.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from .constants import NodePort


class PromptResolver:
    """
    Handles static prompt lookups, chaining, and conjunction templating.
    """

    def __init__(self, nodes: List[Dict], connections: List[Dict]):
        self.nodes_by_id = {n['id']: n for n in nodes}
        self.inputs_by_node = {}
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

        self._prompt_cache = {}

    def get_static_prompt(self, ai_node: Dict) -> str:
        node_id = ai_node['id']
        if node_id in self._prompt_cache:
            return self._prompt_cache[node_id]

        inputs = self.inputs_by_node.get(node_id, [])
        for input_info in inputs:
            if input_info['node']['type'] == 'prompt':
                prompt = input_info['node']['data'].get('text', '')
                self._prompt_cache[node_id] = prompt
                return prompt

        self._prompt_cache[node_id] = ''
        return ''

    def get_prompt_for_image(
        self,
        ai_node: Dict,
        prev_captions: List[str],
        img_index: int,
        base_prompt: str
    ) -> Optional[str]:
        node_id = ai_node['id']
        inputs = self.inputs_by_node.get(node_id, [])

        for input_info in inputs:
            source_node = input_info['node']
            source_type = source_node['type']
            to_port = input_info['to_port']

            if to_port == NodePort.PROMPT_INPUT and source_type == 'aimodel':
                caption = prev_captions[img_index]
                return caption if caption else None

            if source_type == 'conjunction':
                return self._resolve_conjunction(source_node, prev_captions, img_index)

        return base_prompt or None

    def resolve_output(
        self,
        last_ai: Dict,
        output_node: Dict,
        prev_captions: List[str],
        img_index: int
    ) -> str:
        output_id = output_node['id']
        inputs = self.inputs_by_node.get(output_id, [])

        for input_info in inputs:
            source_node = input_info['node']
            if source_node['type'] == 'conjunction':
                conj_inputs = self.inputs_by_node.get(source_node['id'], [])
                if any(inp['node']['id'] == last_ai['id'] for inp in conj_inputs):
                    return self._resolve_conjunction(source_node, prev_captions, img_index)

        return prev_captions[img_index]

    def _resolve_conjunction(self, conj_node: Dict, prev_captions: List[str], img_index: int) -> str:
        template = conj_node['data'].get('template', '')
        if not template:
            return prev_captions[img_index]

        connected_items = conj_node['data'].get('connectedItems', [])
        ref_map = {
            item.get('refKey', ''): (
                prev_captions[img_index]
                if item.get('content', '') == '[Generated Captions]'
                else item.get('content', '')
            )
            for item in connected_items
        }

        resolved = re.sub(
            r'\{([^}]+)\}',
            lambda m: ref_map.get(m.group(1), m.group(0)),
            template
        )

        return resolved
