import os
from flask import Flask, jsonify
from flask_cors import CORS
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
    app: Flask = Flask(__name__)

    # Enable CORS for all routes with specific origins
    CORS(app, resources={r"/api/*": {"origins": "http://localhost"}})

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

    # Add a route for '/'
    @app.route("/", methods=["GET"])
    def root() -> Any:
        """
        Root endpoint that returns a welcome message.

        Returns
        -------
        Any
            JSON response with a welcome message and HTTP status code 200.
        """
        return jsonify({"message": "Welcome to the Model Training API"}), 200

    return app


app: Flask = create_app()
