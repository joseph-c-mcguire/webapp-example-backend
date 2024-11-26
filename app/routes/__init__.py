# app/routes/__init__.py

from .model_serving import model_serving_bp
from .model_training import model_training_bp


def register_routes(app):
    app.register_blueprint(model_serving_bp)
    app.register_blueprint(model_training_bp)
