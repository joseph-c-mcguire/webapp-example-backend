"""Initialize and register route blueprints for the application.

This module sets up the routing for model serving and training functionalities,
as well as helper services and model diagnostics.

"""

from flask import Flask
from .model_serving import model_serving_bp
from .model_training import model_training_bp
from .helper_service import helper_service_bp  # Import the new blueprint
from .model_diagnostics import model_diagnostics_bp  # Import the new blueprint


def register_routes(app: Flask) -> None:
    """
    Register route blueprints with the Flask application.

    Parameters
    ----------
    app : Flask
        The Flask application instance to register the blueprints with.

    """
    app.register_blueprint(model_serving_bp, url_prefix="/")
    app.register_blueprint(model_training_bp)
    app.register_blueprint(helper_service_bp, url_prefix="/api/helper")
    app.register_blueprint(model_diagnostics_bp, url_prefix="/api/diagnostics")
