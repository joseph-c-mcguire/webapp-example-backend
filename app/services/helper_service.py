import os
import logging
import pandas as pd
import json
from flask import jsonify, request
from app.utils.data_utils import load_config
from app.config import Config  # Import Config
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


def get_data() -> Tuple[jsonify, int]:
    """
    Retrieve data from the predictive maintenance CSV file.

    Returns
    -------
    Tuple[jsonify, int]
        JSON response with the data or an error message and HTTP status code.
    """
    logger.info("Starting get_data function")
    config = Config()  # Initialize Config
    data_file_path = config.TEST_DATA_PATH
    logger.debug(f"Data file path: {data_file_path}")
    if not os.path.exists(data_file_path):
        logger.error(f"Data file not found at {data_file_path}")
        return jsonify({"error": "Data file not found"}), 404
    df = pd.read_csv(data_file_path)
    logger.info("Data file loaded successfully")
    data = df.to_dict(orient="records")
    logger.debug(f"Data: {data}")
    return jsonify(data), 200


def get_feature_names() -> jsonify:
    """
    Get the feature names used during training.

    Returns
    -------
    jsonify
        JSON response with the feature names.
    """
    logger.info("Starting get_feature_names function")
    feature_names = [
        "feature1",
        "feature2",
        "feature3",
    ]  # Replace with actual feature names
    logger.info("Getting feature names")
    logger.debug(f"Feature names: {feature_names}")
    return jsonify({"feature_names": feature_names})


def get_model_results() -> Tuple[jsonify, int]:
    """
    Get the results of each model or a specific model.

    Query Parameters
    ----------------
    model_name : str, optional
        Name of the model to query.

    Returns
    -------
    Tuple[jsonify, int]
        JSON response with the results of each model or the specified model and HTTP status code.
    """
    logger.info("Starting get_model_results function")
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
            logger.info(f"Returning results for model: {model_name}")
            return jsonify({model_name: model_results[model_name]}), 200
        else:
            logger.error(f"Model '{model_name}' not found")
            return jsonify({"error": f"Model '{model_name}' not found"}), 404

    logger.info("Returning results for all models")
    return jsonify({"model_results": model_results}), 200


def get_training_progress() -> Tuple[jsonify, int]:
    """
    Get the training progress.

    Returns
    -------
    Tuple[jsonify, int]
        JSON response with the training progress and HTTP status code.
    """
    logger.info("Starting get_training_progress function")
    progress_path = os.path.join(os.path.dirname(__file__), "models", "progress.json")
    logger.debug(f"Progress path: {progress_path}")
    if not os.path.exists(progress_path):
        logger.error("Progress file not found")
        return jsonify({"error": "Progress file not found"}), 404

    with open(progress_path, "r") as f:
        progress = json.load(f)

    logger.info("Returning training progress")
    return jsonify({"progress": progress}), 200


def get_available_models() -> Tuple[jsonify, int]:
    """
    Get the list of available models from the configuration.

    Returns
    -------
    Tuple[jsonify, int]
        JSON response with the list of available models and HTTP status code.
    """
    logger.info("Starting get_available_models function")
    try:
        config = Config()  # Initialize Config
        available_models = list(config.MODEL_PARAMETERS.keys())  # Get model names
        logger.debug(f"Available models: {available_models}")
        return jsonify({"available_models": available_models}), 200
    except Exception as e:
        logger.error(f"Error retrieving available models: {e}")
        return jsonify({"error": str(e)}), 500


def get_class_names() -> Tuple[jsonify, int]:
    """
    Get the class names from the target column in the data.

    Returns
    -------
    Tuple[jsonify, int]
        JSON response with the list of class names and HTTP status code.
    """
    logger.info("Starting get_class_names function")
    try:
        config = Config()  # Initialize Config
        data_file_path = str(config.RAW_DATA_PATH)  # Convert to string
        logger.debug(f"Data file path: {data_file_path}")

        if not os.path.exists(data_file_path):  # Check existence correctly
            logger.error(f"Data file not found at {data_file_path}")
            return jsonify({"error": "Data file not found"}), 404

        df = pd.read_csv(data_file_path)
        logger.info("Data file loaded successfully")
        if isinstance(config.TARGET_COLUMN, list):
            missing_columns = [
                col for col in config.TARGET_COLUMN if col not in df.columns
            ]
            if missing_columns:
                logger.error(f"Columns {missing_columns} not found in the data")
                return (
                    jsonify(
                        {"error": f"Columns {missing_columns} not found in the data"}
                    ),
                    404,
                )
            class_names = []
            for col in config.TARGET_COLUMN:
                class_names.extend(df[col].unique().tolist())
            class_names = list(set(class_names))
        else:
            if config.TARGET_COLUMN not in df.columns:
                logger.error(f"{config.TARGET_COLUMN} column not found in the data")
                return (
                    jsonify(
                        {
                            "error": f"{config.TARGET_COLUMN} column not found in the data"
                        }
                    ),
                    404,
                )
            class_names = df[config.TARGET_COLUMN].unique().tolist()
        logger.debug(f"Class names: {class_names}")
        return jsonify({"class_names": class_names}), 200
    except Exception as e:
        logger.error(f"Error retrieving class names: {e}")
        return jsonify({"error": str(e)}), 500
