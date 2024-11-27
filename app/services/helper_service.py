import os
import logging
import pandas as pd
import json
from flask import jsonify, request
from app.utils.data_utils import load_config
from app.config import Config  # Import Config

logger = logging.getLogger(__name__)


def get_data():
    """
    Retrieve data from the predictive maintenance CSV file.

    Returns
    -------
    flask.Response
        JSON response with the data or an error message.
    """
    config = Config()  # Initialize Config
    data_file_path = config.RAW_DATA_PATH
    logger.debug(f"Data file path: {data_file_path}")
    if not os.path.exists(data_file_path):
        logger.error(f"Data file not found at {data_file_path}")
        return jsonify({"error": "Data file not found"}), 404
    df = pd.read_csv(data_file_path)
    data = df.to_dict(orient="records")
    return jsonify(data)


def get_feature_names():
    """
    Get the feature names used during training.

    Returns
    -------
    flask.Response
        JSON response with the feature names.
    """
    feature_names = [
        "feature1",
        "feature2",
        "feature3",
    ]  # Replace with actual feature names
    logger.info("Getting feature names")
    logger.debug(f"Feature names: {feature_names}")
    return jsonify({"feature_names": feature_names})


def get_model_results():
    """
    Get the results of each model or a specific model.

    Query parameter:
    - model_name: str, optional, name of the model to query

    Returns
    -------
    flask.Response
        JSON response with the results of each model or the specified model.
    """
    results_path = os.path.join(
        os.path.dirname(__file__), "models", "model_results.json"
    )
    logger.debug(f"Results path: {results_path}")
    if not os.path.exists(results_path):
        logger.error(f"Model results file not found at {results_path}")
        return jsonify({"error": "Model results file not found"}), 404

    logger.info(f"Loading model results from {results_path}")
    with open(results_path, "r") as f:
        model_results = json.load(f)

    model_name = request.args.get("model_name")
    if model_name:
        logger.debug(f"Model name: {model_name}")
        if model_name in model_results:
            return jsonify({model_name: model_results[model_name]})
        else:
            return jsonify({"error": f"Model '{model_name}' not found"}), 404

    return jsonify({"model_results": model_results})


def get_training_progress():
    """
    Get the training progress.

    Returns
    -------
    flask.Response
        JSON response with the training progress.
    """
    progress_path = os.path.join(os.path.dirname(__file__), "models", "progress.json")
    logger.debug(f"Progress path: {progress_path}")
    if not os.path.exists(progress_path):
        return jsonify({"error": "Progress file not found"}), 404

    with open(progress_path, "r") as f:
        progress = json.load(f)

    return jsonify({"progress": progress})


def get_available_models():
    """
    Get the list of available models from the configuration.

    Returns
    -------
    flask.Response
        JSON response with the list of available models.
    """
    try:
        config = Config()  # Initialize Config
        available_models = list(config.MODEL_PARAMETERS.keys())  # Get model names
        return jsonify({"available_models": available_models})
    except Exception as e:
        logger.error(f"Error retrieving available models: {e}")
        return jsonify({"error": str(e)}), 500


def get_class_names():
    """
    Get the class names from the target column in the data.

    Returns
    -------
    flask.Response
        JSON response with the list of class names.
    """
    try:
        config = Config()  # Initialize Config
        data_file_path = config.RAW_DATA_PATH
        logger.debug(f"Data file path: {data_file_path}")

        if not data_file_path.exists():
            logger.error(f"Data file not found at {data_file_path}")
            return jsonify({"error": "Data file not found"}), 404

        df = pd.read_csv(data_file_path)
        if config.TARGET_COLUMN not in df.columns:
            logger.error(f"{config.TARGET_COLUMN} column not found in the data")
            return (
                jsonify(
                    {"error": f"{config.TARGET_COLUMN} column not found in the data"}
                ),
                404,
            )

        class_names = df[config.TARGET_COLUMN].unique().tolist()
        return jsonify({"class_names": class_names})
    except Exception as e:
        logger.error(f"Error retrieving class names: {e}")
        return jsonify({"error": str(e)}), 500
