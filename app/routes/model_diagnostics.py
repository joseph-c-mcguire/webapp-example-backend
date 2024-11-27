import logging
from flask import Blueprint, request, jsonify
from app.services.model_diagnostics import (
    get_feature_importance,
    get_roc_curve,
    get_confusion_matrix,
)  # Import the new function
from app.models.model_manager import ModelManager
from app.config import Config
import joblib

model_diagnostics_bp = Blueprint("model_diagnostics", __name__)
logger = logging.getLogger(__name__)


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

    # Load model and preprocessor inside the function
    try:
        config = Config()
        model_manager = ModelManager(config.MODEL_PATH)  # Instantiate model_manager
        model, error = model_manager.load_model(model_name)
        if error:
            return jsonify({"error": error}), 404
        preprocessor_path = config.MODEL_PATH / "preprocessor.pkl"
        logger.info(f"Loading preprocessor from {preprocessor_path}")
        preprocessor = joblib.load(preprocessor_path)
    except Exception as e:
        logger.error(f"Error loading model or preprocessor: {e}")
        return jsonify({"error": "Failed to load model or preprocessor."}), 500

    return get_confusion_matrix(
        model,
        preprocessor,
        model_name,
        class_label_column=config.TARGET_COLUMN,
        target_label=class_label,
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

    try:
        config = Config()
        model_manager = ModelManager(config.MODEL_PATH)  # Instantiate model_manager
        model, error = model_manager.load_model(model_name)
        if error:
            return jsonify({"error": error}), 404
        preprocessor_path = config.MODEL_PATH / "preprocessor.pkl"
        logger.info(f"Loading preprocessor from {preprocessor_path}")
        preprocessor = joblib.load(preprocessor_path)
    except Exception as e:
        logger.error(f"Error loading model or preprocessor: {e}")
        return jsonify({"error": "Failed to load model or preprocessor."}), 500

    return get_roc_curve(
        model,
        preprocessor,
        model_name,
        class_label,
    )


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

    try:
        config = Config()
        model_manager = ModelManager(config.MODEL_PATH)  # Instantiate model_manager
        model, error = model_manager.load_model(model_name)
        if error:
            return jsonify({"error": error}), 404
        # Ensure feature names are retrieved appropriately
        preprocessor_path = config.MODEL_PATH / "preprocessor.pkl"
        preprocessor = joblib.load(preprocessor_path)
        feature_names = preprocessor.get_feature_names_out()
    except Exception as e:
        logger.error(f"Error loading model or preprocessor: {e}")
        return jsonify({"error": "Failed to load model or preprocessor."}), 500

    feature_names = model_manager.get_feature_names()
    return get_feature_importance(model, feature_names)
