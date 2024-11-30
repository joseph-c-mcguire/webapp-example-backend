"""
Module for managing machine learning models, including saving and loading models.
"""

from typing import Tuple, Optional
import joblib
import logging
import os
from pathlib import Path

from sklearn.base import BaseEstimator
from app.config import Config  # Ensure Config is imported

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages saving and loading of machine learning models.

    Attributes
    ----------
    model_path : Path
        The directory path where models are stored, as specified in Config.
    """

    def __init__(self, model_path: Path):
        """
        Initialize the ModelManager with configuration-based model path.

        Parameters
        ----------
        model_path : Path
            The directory path where models are stored.
        """
        self.config = Config()  # Singleton instance
        self.model_path = self.config.MODEL_PATH  # Use Config for model path

    def load_model(self, model_name: str) -> Tuple[Optional[BaseEstimator], str]:
        """
        Load a machine learning model by its name.

        Parameters
        ----------
        model_name : str
            The name of the model to be loaded.

        Returns
        -------
        Tuple[Optional[BaseEstimator], str]
            A tuple containing the loaded model and an error message if loading failed.
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

    def save_model(self, model: BaseEstimator, model_name: str) -> None:
        """
        Save a machine learning model to the specified file path.

        Parameters
        ----------
        model : BaseEstimator
            The machine learning model object to save.
        model_name : str
            The name of the model file to save.
        """
        save_path = self.model_path / (
            model_name if model_name.endswith(".pkl") else f"{model_name}.pkl"
        )
        logger.info(f"Saving model to {save_path}")
        joblib.dump(model, save_path)
        logger.info("Model saved successfully")
