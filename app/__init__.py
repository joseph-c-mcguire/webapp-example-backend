import os

from flask import Flask
from flask_cors import CORS
from app.config import DevelopmentConfig, ProductionConfig, TestingConfig
from app.routes import register_routes
from app.extensions import cors


def create_app():
    app = Flask(__name__)
    CORS(app)  # Enable CORS

    # Load configuration based on environment
    env = os.getenv("FLASK_ENV", "development")
    if env == "development":
        app.config.from_object(DevelopmentConfig)
    elif env == "testing":
        app.config.from_object(TestingConfig)
    else:
        app.config.from_object(ProductionConfig)

    # Register routes
    register_routes(app)

    # Initialize extensions
    cors.init_app(app)

    return app
