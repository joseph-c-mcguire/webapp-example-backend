import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import joblib
from flask import Blueprint, jsonify, request

from app.config import Config  # Ensure Config is imported
from app.models.model_manager import ModelManager
from app.services.inference_service import InferenceService  # Ensure proper import

model_serving_bp = Blueprint("model_serving", __name__)
logger = logging.getLogger(__name__)

config = Config()  # Singleton instance
model_manager = ModelManager(config.MODEL_PATH)

# Initialize InferenceService
preprocessor_path = config.MODEL_PATH / "preprocessor.pkl"  # Updated path
preprocessor = joblib.load(preprocessor_path)
inference_service = InferenceService(
    preprocessor=preprocessor, model_path=config.MODEL_PATH
)  # Updated instantiation


@model_serving_bp.route("/predict-probabilities", methods=["POST"])
def predict_probabilities() -> Union[Tuple[Dict[str, Any], int], Dict[str, Any]]:
    """
    Predict probabilities for equipment failure types using the specified model.

    Parameters
    ----------
    None

    Returns
    -------
    Union[Tuple[Dict[str, Any], int], Dict[str, Any]]
        - On success: A dictionary containing prediction probabilities.
        - On error: A dictionary with an error message and the corresponding HTTP status code.
    """
    try:
        data = request.get_json(
            force=False, silent=True
        )  # Do not force, use silent parsing
        if data is None:
            model_name = "Decision Tree"
            features = [
                {
                    "UDI": 1,
                    "Product ID": "M14860",
                    "Type": "M",
                    "Air temperature [K]": 298.1,
                    "Process temperature [K]": 308.6,
                    "Rotational speed [rpm]": 1551,
                    "Torque [Nm]": 42.8,
                    "Tool wear [min]": 0,
                    "Target": 0,
                    "Failure Type": "No Failure",
                }
            ]
        else:
            model_name = data.get("model_name", "Decision Tree")
            features = data.get(
                "data",
                [
                    {
                        "UDI": 1,
                        "Product ID": "M14860",
                        "Type": "M",
                        "Air temperature [K]": 298.1,
                        "Process temperature [K]": 308.6,
                        "Rotational speed [rpm]": 1551,
                        "Torque [Nm]": 42.8,
                        "Tool wear [min]": 0,
                        "Target": 0,
                        "Failure Type": "No Failure",
                    }
                ],
            )  # Default features as list of dicts
        logger.debug(f"Model name: {model_name}")
        logger.debug(f"Input data: {features}")

        # Use InferenceService for prediction
        prediction_result = inference_service.predict_probabilities(
            model_name, features
        )

        if prediction_result is None:
            logger.error("InferenceService returned None")
            return jsonify({"error": "Prediction failed due to internal error."}), 500

        if "error" in prediction_result:
            logger.error(f"Prediction error: {prediction_result['error']}")
            return jsonify({"error": prediction_result["error"]}), 400

        return jsonify({"probabilities": prediction_result["probabilities"]})
    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        return jsonify({"error": str(e)}), 400
