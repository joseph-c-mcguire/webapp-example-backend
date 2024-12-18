import logging
import os
from pathlib import Path
from flask import Blueprint, jsonify, Response
from app.utils.data_preprocessing import split_data
from app.services.training_service import TrainingService
from app.config import Config

model_training_bp = Blueprint("model_training", __name__)
logger = logging.getLogger(__name__)


@model_training_bp.route("/train", methods=["POST"])
def train_model() -> Response:
    """
    Train the model by calling the train_model.py script.

    Returns
    -------
    flask.Response
        JSON response with the result of the training process.
    """
    config = Config()  # Singleton instance
    data_path = Path(config.RAW_DATA_PATH)  # Convert to Path
    processed_data_path = Path(config.PROCESSED_DATA_PATH)  # Convert to Path
    train_val_file_path = processed_data_path / "train_val_data.csv"
    test_file_path = processed_data_path / "test_data.csv"
    logger.debug(f"Train/validation file path: {train_val_file_path}")
    logger.debug(f"Test file path: {test_file_path}")

    # Check if the train and test data files exist
    if not os.path.exists(train_val_file_path) or not os.path.exists(test_file_path):
        logger.info(
            "Train or test data file not found. Running split_data function to generate them."
        )
        try:
            split_data(
                data_file_path=data_path,
                train_val_file_path=train_val_file_path,
                test_file_path=test_file_path,
            )
        except Exception as e:
            logger.error(f"An error occurred while splitting the data: {str(e)}")
            return (
                jsonify(
                    {
                        "error": "An error occurred while splitting the data",
                        "details": str(e),
                    },
                ),
                500,
            )

    # Proceed with training the model using TrainingService
    try:
        logger.info("Starting model training")
        training_service = TrainingService(config_path=config.CONFIG_PATH)
        result = training_service.train_model()
        return jsonify({"message": "Model training completed successfully"}), 200
    except Exception as e:
        logger.error(f"An error occurred while training the model: {str(e)}")
        return (
            jsonify(
                {
                    "error": "An error occurred while training the model",
                    "details": str(e),
                },
            ),
            500,
        )
