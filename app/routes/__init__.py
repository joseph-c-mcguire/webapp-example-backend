"""Initialize and register route blueprints for the application.

This module sets up the routing for model serving and training functionalities.

"""

from .model_serving import model_serving_bp
from .model_training import model_training_bp
from .helper_service import helper_service_bp  # Import the new blueprint
from .model_diagnostics import model_diagnostics_bp  # Import the new blueprint


def register_routes(app):
    app.register_blueprint(model_serving_bp)
    app.register_blueprint(model_training_bp)
    app.register_blueprint(helper_service_bp)  # Register the new blueprint
    app.register_blueprint(model_diagnostics_bp)  # Register the new blueprint
