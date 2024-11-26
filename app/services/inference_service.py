import logging
from typing import Dict, List, Any
from pathlib import Path
from pandas import DataFrame
from sklearn.base import BaseEstimator
from app.models.model_manager import ModelManager
from app.config import Config  # Ensure Config is imported

logger = logging.getLogger(__name__)


class InferenceService:
    """
    Service for making predictions and predicting probabilities using specified models.

    Attributes:
        preprocessor (BaseEstimator): A preprocessor for transforming input features.
    """

    def __init__(self, preprocessor: BaseEstimator, model_path: Path):
        """
        Initialize the InferenceService with the given preprocessor and model path.

        Parameters:
            preprocessor (BaseEstimator): A preprocessor for transforming input features.
            model_path (Path): Path to the directory containing models.
        """
        self.preprocessor = preprocessor
        self.config = Config()  # Singleton instance
        self.model_manager = ModelManager(model_path=self.config.MODEL_PATH)

    def predict(self, model_name: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict equipment failure type using the specified model.

        Parameters:
            model_name (str): Name of the model to use for prediction.
            features (Dict[str, Any]): Input features for prediction.

        Returns:
            Dict[str, Any]: Prediction result.
        """
        model, error = self.model_manager.load_model(model_name)  # Updated method call
        if error:
            return {"error": error}

        try:
            features_df = DataFrame(features, index=[0])
            logger.info(f"Predicting with model: {model_name}")
            logger.info(f"Predicting with data: {features_df}")
            features_transformed = self.preprocessor.transform(features_df)
            logger.info(f"Transformed features: {features_transformed}")
            prediction = model.predict(features_transformed)
            return {"prediction": prediction[0]}
        except ValueError as e:
            logger.error(f"Error during prediction: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {"error": str(e)}

    def predict_probabilities(
        self, model_name: str, data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Predict probabilities for equipment failure type using the specified model.

        Parameters:
            model_name (str): Name of the model to use for prediction.
            data (List[Dict[str, Any]]): Input data for prediction.

        Returns:
            Dict[str, Any]: Prediction probabilities or error message.
        """
        return self._predict_probabilities(model_name, data)

    def _predict_probabilities(
        self, model_name: str, data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Internal method to handle probability predictions.
        """
        model, error = self.model_manager.load_model(model_name)  # Updated method call
        if error:
            return {"error": error}

        try:
            if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
                logger.error("Data should be a list of dictionaries.")
                return {
                    "error": "Invalid data format. Expected a list of dictionaries."
                }

            features_df = DataFrame(data)
            features_transformed = self.preprocessor.transform(features_df)
            logger.debug(f"Transformed features: {features_transformed}")

            if not hasattr(model, "predict_proba"):
                logger.error(
                    f"Model {model_name} does not support probability prediction"
                )
                return {
                    "error": f"Model {model_name} does not support probability prediction"
                }

            probabilities = model.predict_proba(
                features_transformed
            )  # Multi-class probabilities
            return {"probabilities": probabilities.tolist()}
        except ValueError as e:
            logger.error(f"Error during prediction: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {"error": str(e)}
