import logging
from flask import Blueprint, request, jsonify
from app.services.model_diagnostics import (
    get_feature_importance,
    get_roc_curve,
    get_confusion_matrix,
)  # Import the new function
from app.models.model_manager import ModelManager
from app.config import Config

model_diagnostics_bp = Blueprint("model_diagnostics", __name__)
logger = logging.getLogger(__name__)

config = Config()  # Singleton instance
model_manager = ModelManager(config.MODEL_PATH)


@model_diagnostics_bp.route("/confusion-matrix", methods=["POST"])
def confusion_matrix_endpoint():
    """
    Get the confusion matrix for the specified model using test data from test_data.csv.

    Request JSON format:
    {
        "model_name": "LogisticRegression",
        "class_label": "class_name"
    }

    Returns
    -------
    flask.Response
        JSON response with the confusion matrix.
    """
    data = request.get_json(force=True)
    model_name = data.get("model_name", "Logistic Regression")
    class_label = data.get("class_label", "No Failure")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Class label: {class_label}")

    model, error = model_manager.load_model(model_name)
    if error:
        return jsonify({"error": error}), 404

    return get_confusion_matrix(
        model, model_manager.preprocessor, model_name, class_label
    )


@model_diagnostics_bp.route("/roc-curve", methods=["POST"])
def roc_curve_endpoint():
    """
    Generate the ROC curve data for the specified model using test data from test_data.csv.

    Request JSON format:
    {
        "model_name": "LogisticRegression",
        "class_label": "class_name"
    }

    Returns
    -------
    flask.Response
        JSON response with the false positive rate, true positive rate, and AUC.
    """
    data = request.get_json(force=True)
    model_name = data.get("model_name", "Logistic Regression")
    class_label = data.get("class_label", "No Failure")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Class label: {class_label}")

    model, error = model_manager.load_model(model_name)
    if error:
        return jsonify({"error": error}), 404

    return get_roc_curve(model, model_manager.preprocessor, model_name, class_label)


@model_diagnostics_bp.route("/feature-importance", methods=["GET"])
def feature_importance_endpoint():
    """
    Get the feature importance of the specified model.

    Query parameter:
    - model_name: str, optional, name of the model to query

    Returns
    -------
    flask.Response
        JSON response with the feature importance.
    """
    model_name = request.args.get("model_name", "Gradient Boosting")
    logger.debug(f"Model name: {model_name}")

    model, error = model_manager.load_model(model_name)
    if error:
        return jsonify({"error": error}), 404

    feature_names = model_manager.get_feature_names()
    return get_feature_importance(model, feature_names)
