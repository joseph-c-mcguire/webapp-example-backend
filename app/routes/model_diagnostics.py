import logging
from flask import Blueprint, request, jsonify
import pandas as pd
from app.services.model_diagnostics import (
    get_feature_importance,
    get_roc_curve,
    get_confusion_matrix,
)
from app.models.model_manager import ModelManager
from app.config import Config
import joblib
from typing import Tuple, Dict, Any

model_diagnostics_bp = Blueprint("model_diagnostics", __name__)
logger = logging.getLogger(__name__)

config = Config()  # Add this line to instantiate Config


@model_diagnostics_bp.route("/confusion-matrix", methods=["POST"])
def confusion_matrix_endpoint() -> Tuple[Dict[str, Any], int]:
    """
    Get the confusion matrix for the specified model using test data from test_data.csv.

    Parameters
    ----------
    None

    Request JSON format
    -------------------
    {
        "model_name": str, optional, default="Logistic Regression"
        "class_label": str, optional, default="No Failure"
    }

    Returns
    -------
    flask.Response
        JSON response with the confusion matrix.
    """
    data = request.get_json(force=True)
    model_name: str = data.get("model_name", "Logistic Regression")
    class_label: str = data.get("class_label", "No Failure")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Class label: {class_label}")

    # Load model and preprocessor inside the function
    try:
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
def roc_curve_endpoint() -> Tuple[Dict[str, Any], int]:
    """
    Generate the ROC curve data for the specified model using test data from test_data.csv.

    Parameters
    ----------
    None

    Request JSON format
    -------------------
    {
        "model_name": str, optional, default="Decision Tree"
        "class_label": str, optional, default="No Failure"
    }

    Returns
    -------
    flask.Response
        JSON response with the false positive rate, true positive rate, and AUC.
    """
    try:
        data = request.get_json(silent=True)
        if data is None:
            logger.error("Invalid or missing JSON data")
            return jsonify({"error": "Invalid or missing JSON data"}), 400
        logger.debug(f"Received JSON data: {data}")
        model_name: str = data.get("model_name", "Decision Tree")
        class_label: str = data.get("class_label", "No Failure")
    except Exception as e:
        logger.error(f"Error parsing JSON data: {e}")
        return jsonify({"error": "Invalid or missing JSON data"}), 400

    logger.info(f"Model name: {model_name}")
    logger.info(f"Class label: {class_label}")

    try:
        model_manager = ModelManager(config.MODEL_PATH)  # Instantiate model_manager
        model, error = model_manager.load_model(model_name)
        if error:
            return jsonify({"error": error}), 404
        preprocessor_path = config.MODEL_PATH / "preprocessor.pkl"
        test_data_path = config.TEST_DATA_PATH  # Added line
        if not test_data_path.exists():
            logger.error(f"Test data file not found at {test_data_path}")
            return (
                jsonify(
                    {
                        "error": "Test data file not found. Please ensure 'test_data.csv' exists in the 'data/processed' directory."
                    }
                ),
                404,
            )
        test_data = pd.read_csv(test_data_path)  # Added line to load test data
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
def feature_importance_endpoint() -> Tuple[Dict[str, Any], int]:
    """
    Get the feature importance of the specified model.

    Query Parameters
    ----------------
    model_name : str, optional, default="Gradient Boosting"
        Name of the model to query.

    Returns
    -------
    flask.Response
        JSON response with the feature importance.
    """
    model_name: str = request.args.get("model_name", "Gradient Boosting")
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

    return get_feature_importance(model, feature_names)
