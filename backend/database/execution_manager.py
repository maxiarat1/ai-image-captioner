"""
Execution Manager - Persistence layer for graph execution jobs.

Handles database operations for tracking execution jobs that persist
across browser sessions and page refreshes.
"""

import duckdb
import uuid
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any

from config import DATABASE_PATH

logger = logging.getLogger(__name__)


class ExecutionManager:
    """Manages execution job persistence in DuckDB."""

    def __init__(self):
        # Use separate database file for execution jobs to avoid conflicts
        main_db_path = Path(DATABASE_PATH)
        self.db_path = main_db_path.parent / "execution_jobs.duckdb"
        self._init_database()
        logger.info(f"Execution manager initialized with database: {self.db_path}")

    def _init_database(self) -> None:
        """Create execution_jobs table if not exists."""
        conn = self._get_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS execution_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    graph_json TEXT NOT NULL,
                    image_ids_json TEXT NOT NULL,
                    current_stage INTEGER DEFAULT 0,
                    total_stages INTEGER DEFAULT 1,
                    processed INTEGER DEFAULT 0,
                    total INTEGER DEFAULT 0,
                    success INTEGER DEFAULT 0,
                    failed INTEGER DEFAULT 0,
                    progress_json TEXT,
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Get new database connection.

        DuckDB handles concurrency internally, each connection
        should be used from a single thread.
        """
        return duckdb.connect(str(self.db_path), read_only=False)

    def create_job(self, graph: Dict[str, Any], image_ids: List[str]) -> str:
        """
        Create new execution job and return job_id.

        Args:
            graph: Node graph structure (nodes + connections)
            image_ids: List of image IDs to process

        Returns:
            job_id: Unique job identifier
        """
        job_id = str(uuid.uuid4())
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO execution_jobs
                (job_id, status, graph_json, image_ids_json, total)
                VALUES (?, ?, ?, ?, ?)
            """, (job_id, 'pending', json.dumps(graph), json.dumps(image_ids), len(image_ids)))
            conn.commit()
            logger.info(f"Created job {job_id} with {len(image_ids)} images")
            return job_id
        except Exception as e:
            logger.error(f"Failed to create job: {e}")
            raise
        finally:
            conn.close()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job by ID with full details including graph.

        Args:
            job_id: Job identifier

        Returns:
            Job dict with all fields or None if not found
        """
        conn = self._get_connection()
        try:
            result = conn.execute("""
                SELECT job_id, status, graph_json, image_ids_json, current_stage, total_stages,
                       processed, total, success, failed, progress_json, error,
                       created_at, updated_at
                FROM execution_jobs WHERE job_id = ?
            """, (job_id,)).fetchone()

            if not result:
                return None

            return {
                'job_id': result[0],
                'status': result[1],
                'graph': json.loads(result[2]),
                'image_ids': json.loads(result[3]),
                'current_stage': result[4],
                'total_stages': result[5],
                'processed': result[6],
                'total': result[7],
                'success': result[8],
                'failed': result[9],
                'progress': json.loads(result[10]) if result[10] else {},
                'error': result[11],
                'created_at': str(result[12]),
                'updated_at': str(result[13])
            }
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            return None
        finally:
            conn.close()

    def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job status (lightweight, without graph data).

        Args:
            job_id: Job identifier

        Returns:
            Status dict or None if not found
        """
        conn = self._get_connection()
        try:
            result = conn.execute("""
                SELECT job_id, status, current_stage, total_stages,
                       processed, total, success, failed, progress_json, error
                FROM execution_jobs WHERE job_id = ?
            """, (job_id,)).fetchone()

            if not result:
                return None

            return {
                'job_id': result[0],
                'status': result[1],
                'current_stage': result[2],
                'total_stages': result[3],
                'processed': result[4],
                'total': result[5],
                'success': result[6],
                'failed': result[7],
                'progress': json.loads(result[8]) if result[8] else {},
                'error': result[9]
            }
        except Exception as e:
            logger.error(f"Failed to get status for job {job_id}: {e}")
            return None
        finally:
            conn.close()

    def update_status(
        self,
        job_id: str,
        status: Optional[str] = None,
        current_stage: Optional[int] = None,
        total_stages: Optional[int] = None,
        processed: Optional[int] = None,
        success: Optional[int] = None,
        failed: Optional[int] = None,
        progress: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> bool:
        """
        Update job status fields.

        Args:
            job_id: Job identifier
            status: Job status ('pending', 'running', 'completed', 'failed', 'cancelled')
            current_stage: Current processing stage number
            total_stages: Total number of stages
            processed: Number of images processed
            success: Number of successful generations
            failed: Number of failed generations
            progress: Progress metadata (speed, ETA, etc.)
            error: Error message if failed

        Returns:
            True if updated successfully
        """
        # Build update fields dynamically
        fields = {
            'status': status,
            'current_stage': current_stage,
            'total_stages': total_stages,
            'processed': processed,
            'success': success,
            'failed': failed,
            'progress_json': json.dumps(progress) if progress is not None else None,
            'error': error
        }

        # Filter out None values
        updates = [f"{k} = ?" for k, v in fields.items() if v is not None]
        params = [v for v in fields.values() if v is not None]

        if not updates:
            return False

        # Always update timestamp
        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(job_id)

        conn = self._get_connection()
        try:
            query = f"UPDATE execution_jobs SET {', '.join(updates)} WHERE job_id = ?"
            conn.execute(query, params)
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to update job {job_id}: {e}")
            return False
        finally:
            conn.close()

    def cancel_job(self, job_id: str) -> bool:
        """
        Mark job as cancelled.

        Args:
            job_id: Job identifier

        Returns:
            True if cancelled successfully
        """
        return self.update_status(job_id, status='cancelled')

    def delete_job(self, job_id: str) -> bool:
        """
        Delete job from database.

        Args:
            job_id: Job identifier

        Returns:
            True if deleted successfully
        """
        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM execution_jobs WHERE job_id = ?", (job_id,))
            conn.commit()
            logger.info(f"Deleted job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete job {job_id}: {e}")
            return False
        finally:
            conn.close()

    def cleanup_old_jobs(self, days: int = 7) -> int:
        """
        Clean up completed/failed/cancelled jobs older than specified days.

        Args:
            days: Number of days to retain jobs (default: 7)

        Returns:
            Number of jobs deleted
        """
        conn = self._get_connection()
        try:
            # DuckDB uses INTERVAL syntax
            result = conn.execute("""
                DELETE FROM execution_jobs
                WHERE status IN ('completed', 'failed', 'cancelled')
                AND created_at < NOW() - INTERVAL ? DAY
            """, (days,))
            deleted_count = result.fetchall()[0][0] if result else 0
            conn.commit()
            logger.info(f"Cleaned up {deleted_count} old jobs (older than {days} days)")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup old jobs: {e}")
            return 0
        finally:
            conn.close()

    def list_active_jobs(self) -> List[Dict[str, Any]]:
        """
        List all active (running/pending) jobs.

        Returns:
            List of job status dicts
        """
        conn = self._get_connection()
        try:
            results = conn.execute("""
                SELECT job_id, status, current_stage, total_stages,
                       processed, total, success, failed, created_at
                FROM execution_jobs
                WHERE status IN ('pending', 'running')
                ORDER BY created_at DESC
            """).fetchall()

            jobs = []
            for row in results:
                jobs.append({
                    'job_id': row[0],
                    'status': row[1],
                    'current_stage': row[2],
                    'total_stages': row[3],
                    'processed': row[4],
                    'total': row[5],
                    'success': row[6],
                    'failed': row[7],
                    'created_at': str(row[8])
                })
            return jobs
        except Exception as e:
            logger.error(f"Failed to list active jobs: {e}")
            return []
        finally:
            conn.close()
