"""
Simple in-memory store for node outputs during graph execution.
"""

from collections import defaultdict
from typing import Dict, Iterable


class GraphDataStore:
    """
    Stores node outputs keyed by (node_id, port_index).

    Each entry is a mapping of image_id -> value, keeping the per-image
    association explicit so downstream nodes can merge information.
    """

    def __init__(self, image_ids: Iterable[str]):
        self.image_ids = list(image_ids)
        self._storage: Dict[str, Dict[int, Dict[str, object]]] = defaultdict(dict)

    def set_output(self, node_id: str, port_index: int, values: Dict[str, object]) -> None:
        port_storage = self._storage[node_id].setdefault(port_index, {})
        port_storage.update(values)

    def get_output(self, node_id: str, port_index: int) -> Dict[str, object]:
        return self._storage.get(node_id, {}).get(port_index, {})

    def get_latest_text(self) -> Dict[str, str]:
        """
        Convenience helper returning the latest text stored for each image.

        Many nodes treat "text" outputs identically regardless of origin.
        """
        latest: Dict[str, str] = {img_id: "" for img_id in self.image_ids}
        for node_outputs in self._storage.values():
            for values in node_outputs.values():
                for img_id, value in values.items():
                    if isinstance(value, str):
                        latest[img_id] = value
        return latest
