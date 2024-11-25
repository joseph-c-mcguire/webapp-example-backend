from flask import Blueprint, request, jsonify
from pandas import DataFrame
import logging
from app.utils.model_utils import load_specific_model

bp = Blueprint("model_serving", __name__)
logger = logging.getLogger(__name__)


@bp.route("/predict", methods=["POST"])
def predict():
    """
    Predict equipment failure type using the specified model.

    Request JSON format:
    {
        "model_name": "DecisionTree",
        "features": {
            "Type": "M",
            "Air temperature [K]": 300,
            "Process temperature [K]": 310,
            "Rotational speed [rpm]": 1500,
            "Torque [Nm]": 40,
            "Tool wear [min]": 10
        }
    }

    Returns:
    JSON response with prediction result.
    """
    data = request.get_json(force=True)
    logger.info(f"Received data: {data}")
    model_name = data.get("model_name", "Decision Tree")
    logger.info(f"Model name: {model_name}")
    model, error = load_specific_model(model_name)
    if error:
        return jsonify({"error": error}), 404

    try:
        features = DataFrame(data["features"], index=[0])
        logger.info(f"Predicting with model: {model_name}")
        logger.info(f"Predicting with data: {features}")
        features = preprocessor.transform(features)
        logger.info(f"Transformed features: {features}")
        prediction = model.predict(features)
        return jsonify({"prediction": prediction[0]})
    except ValueError as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400


@bp.route("/predict-probabilities", methods=["POST"])
def predict_probabilities():
    """
    Predict probabilities for equipment failure type using the specified model.

    Request JSON format:
    {
        "model_name": "DecisionTree",
        "data": [
            {
                "Type": "M",
                "Air temperature [K]": 300,
                "Process temperature [K]": 310,
                "Rotational speed [rpm]": 1500,
                "Torque [Nm]": 40,
                "Tool wear [min]": 10
            }
        ]
    }

    Returns:
    JSON response with prediction probabilities.
    """
    data = request.get_json(force=True)
    model_name = data.get("model_name", "Decision Tree")
    logger.debug(f"Model name: {model_name}")
    model, error = load_specific_model(model_name)
    if error:
        return jsonify({"error": error}), 404

    try:
        features = DataFrame(data["data"])
        features = preprocessor.transform(features)
        logger.debug(f"Transformed features: {features}")

        if not hasattr(model, "predict_proba"):
            logger.error(f"Model {model_name} does not support probability prediction")
            return (
                jsonify(
                    {
                        "error": f"Model {model_name} does not support probability prediction"
                    }
                ),
                400,
            )

        probabilities = model.predict_proba(features)  # Multi-class probabilities
        return jsonify({"probabilities": probabilities.tolist()})
    except ValueError as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400
