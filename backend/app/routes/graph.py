"""
Graph execution routes for node-based processing.
"""
import json
import asyncio
import threading
import logging
from flask import Blueprint, request, Response, stream_with_context
from graph_executor import GraphExecutor
from app.utils.route_decorators import handle_route_errors

logger = logging.getLogger(__name__)

bp = Blueprint('graph', __name__)


def init_routes(model_manager, session_manager, async_session_manager, execution_manager, active_executors):
    """Initialize routes with dependencies."""

    @bp.route('/graph/execute', methods=['POST'])
    @handle_route_errors("executing graph")
    def execute_graph():
        """Submit a graph for execution."""
        data = request.get_json()
        graph = data.get('graph', {})
        image_ids = data.get('image_ids', [])
        clear_previous = data.get('clear_previous', True)

        if not graph or not image_ids:
            raise ValueError("Missing graph or image_ids")

        # Clear previous captions if requested
        if clear_previous:
            session_manager.clear_all_captions()
            logger.info("Cleared previous captions before execution")

        # Create job
        job_id = execution_manager.create_job(graph, image_ids)

        # Start execution in background thread
        executor = GraphExecutor(execution_manager, async_session_manager)
        active_executors[job_id] = executor

        def run_executor():
            try:
                asyncio.run(executor.execute(job_id, model_manager.get_model))
            finally:
                active_executors.pop(job_id, None)

        thread = threading.Thread(target=run_executor, daemon=True)
        thread.start()

        logger.info(f"Started execution job {job_id} in background")

        return {
            "success": True,
            "job_id": job_id,
            "message": "Graph execution started"
        }

    @bp.route('/graph/status/<job_id>', methods=['GET'])
    @handle_route_errors("getting graph status")
    def get_graph_status_sse(job_id):
        """Get real-time status updates for a job via Server-Sent Events."""
        def generate():
            import time
            last_status = None

            while True:
                status = execution_manager.get_status(job_id)

                if not status:
                    yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                    break

                # Only send if status changed
                if status != last_status:
                    yield f"data: {json.dumps(status)}\n\n"
                    last_status = status

                # If terminal state, close connection
                if status['status'] in ('completed', 'failed', 'cancelled'):
                    break

                time.sleep(0.5)  # Poll every 500ms

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )

    @bp.route('/graph/cancel/<job_id>', methods=['POST'])
    @handle_route_errors("cancelling graph execution")
    def cancel_graph_execution(job_id):
        """Cancel a running graph execution."""
        # Signal executor to stop
        executor = active_executors.get(job_id)
        if executor:
            executor.cancel()

        # Update database status
        execution_manager.cancel_job(job_id)

        logger.info(f"Cancelled job {job_id}")

        return {
            "success": True,
            "message": "Job cancellation requested"
        }

    return bp
