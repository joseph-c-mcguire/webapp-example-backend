import os
from flask import Flask
from app.config import DevelopmentConfig, ProductionConfig, TestingConfig
from app.routes import register_routes
from app.extensions import init_app as init_extensions
from typing import Any


def create_app() -> Flask:
    """
    Create and configure the Flask application.

    Returns
    -------
    Flask
        The configured Flask application instance.
    """
    app = Flask(__name__)

    # Load configuration based on environment
    env: str = os.getenv("FLASK_ENV", "development")
    if env == "development":
        app.config.from_object(DevelopmentConfig)
    elif env == "testing":
        app.config.from_object(TestingConfig)
    else:
        app.config.from_object(ProductionConfig)

    # Register routes
    register_routes(app)

    # Initialize extensions
    init_extensions(app)

    # Register blueprints
    from app.routes.model_training import model_training_bp
    from app.routes.model_serving import model_serving_bp
    from app.routes.model_diagnostics import model_diagnostics_bp
    from app.routes.helper_service import helper_service_bp

    app.register_blueprint(model_training_bp)
    app.register_blueprint(model_serving_bp)
    app.register_blueprint(model_diagnostics_bp)
    app.register_blueprint(helper_service_bp)

    # Add a route for '/'
    @app.route("/")
    def index():
        return "Welcome to the Machine Learning API"

    return app
