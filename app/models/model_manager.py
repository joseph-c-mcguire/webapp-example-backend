from typing import Tuple
import joblib
import logging
import os
from pathlib import Path

from sklearn.base import BaseEstimator
from app.config import Config  # Ensure Config is imported

logger = logging.getLogger(__name__)


class ModelManager:
    """
    A class to manage saving and loading machine learning models.

    Attributes:
        model_path (Path): The directory path where models are stored.
    """

    def __init__(self, model_path: Path):
        """
        Initializes the ModelManager with the given model path.

        Args:
            model_path (Path): The directory path where models are stored.
        """
        self.config = Config()  # Singleton instance
        self.model_path = self.config.MODEL_PATH  # Use Config for model path

    def load_model(self, model_name: str) -> Tuple[BaseEstimator, str]:
        """
        Load a specific model by name.

        Parameters:
            model_name (str): The name of the model to load.

        Returns:
            Tuple[BaseEstimator, str]: The loaded model and an error message if any.
        """
        model_path = self.model_path / f"{model_name}.pkl"
        if not model_path.exists():
            logger.error(f"Model file not found at {model_path}")
            return None, f"Model '{model_name}' not found"
        try:
            logger.info(f"Loading model from {model_path}")
            model = joblib.load(model_path)
            return model, ""
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None, str(e)

    @staticmethod
    def save_model(model, file_path):
        """
        Saves a model to the specified file path.

        Args:
            model: The model object to save.
            file_path (str): The file path where the model will be saved.
        """
        logger.info(f"Saving model to {file_path}")
        joblib.dump(model, file_path)
        logger.info("Model saved successfully")
