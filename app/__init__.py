import os
from flask import Flask, jsonify
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

    # Import blueprints inside the function to avoid circular imports
    from app.routes.model_training import model_training_bp
    from app.routes.model_serving import model_serving_bp
    from app.routes.model_diagnostics import model_diagnostics_bp
    from app.routes.helper_service import helper_service_bp

    # Register blueprints with unique names and URL prefixes
    app.register_blueprint(
        model_training_bp, name="training_bp", url_prefix="/api/training"
    )
    app.register_blueprint(
        model_serving_bp, name="serving_bp", url_prefix="/api/serving"
    )
    app.register_blueprint(
        model_diagnostics_bp, name="diagnostics_bp", url_prefix="/api/diagnostics"
    )
    app.register_blueprint(
        helper_service_bp, name="helper_bp", url_prefix="/api/helper"
    )

    # Add a route for '/'
    @app.route("/", methods=["GET"])
    def root():
        return jsonify({"message": "Welcome to the Model Training API"}), 200

    return app


app = create_app()
