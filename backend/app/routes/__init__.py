"""
Route blueprints registration.
"""
from . import models, sessions, captions, export, graph


def register_blueprints(app, model_manager, session_manager, flow_control_hub,
                       execution_manager, active_executors, categories, model_metadata):
    """Register all route blueprints with the Flask app."""

    # Register models routes
    models_bp = models.init_routes(model_manager, categories, model_metadata)
    app.register_blueprint(models_bp)

    # Register sessions routes
    sessions_bp = sessions.init_routes(session_manager)
    app.register_blueprint(sessions_bp)

    # Register captions routes (now uses Flow Control Hub)
    captions_bp = captions.init_routes(model_manager, flow_control_hub)
    app.register_blueprint(captions_bp)

    # Register export routes
    export_bp = export.init_routes()
    app.register_blueprint(export_bp)

    # Register graph routes (now uses Flow Control Hub)
    graph_bp = graph.init_routes(model_manager, session_manager, flow_control_hub,
                                  execution_manager, active_executors)
    app.register_blueprint(graph_bp)
