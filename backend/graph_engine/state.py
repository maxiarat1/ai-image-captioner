"""
Execution state tracking for graph jobs.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ExecutionState:
    """
    Lightweight execution tracker that mirrors job status to the database.

    This class intentionally stays dumb; it simply keeps a few counters and
    delegates persistence to ExecutionManager.
    """

    exec_manager: "ExecutionManager"
    job_id: str
    total_images: int

    def __post_init__(self):
        self.start_time = time.time()
        self.current_node_processed = 0
        self.success = 0
        self.failed = 0
        self.current_node_id = None

    def set_current_node(self, node_id: str) -> None:
        self.current_node_id = node_id
        self.current_node_processed = 0

    def increment_progress(self) -> None:
        self.current_node_processed += 1

    def increment(self, success: bool = True) -> None:
        if success:
            self.success += 1
        else:
            self.failed += 1

    def update(self, extra_progress: Dict[str, Any] | None = None) -> None:
        elapsed = time.time() - self.start_time
        speed = self.current_node_processed / elapsed if elapsed > 0 else 0

        progress_payload = {
            'speed': f"{speed:.1f} img/s" if speed > 0 else "",
            'progress': f"{self.current_node_processed}/{self.total_images}",
            'current_node_id': self.current_node_id
        }
        if extra_progress:
            progress_payload.update(extra_progress)

        self.exec_manager.update_status(
            self.job_id,
            current_stage=1,
            total_stages=1,
            processed=self.current_node_processed,
            success=self.success,
            failed=self.failed,
            progress=progress_payload
        )
