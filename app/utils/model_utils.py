import logging
import joblib

from pandas import to_pickle

logger = logging.getLogger(__name__)


def load_specific_model(model_name):
    model_path = MODEL_PATH / f"{model_name}.pkl"
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        return None, f"Model '{model_name}' not found"
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    return model, None


def save_model(model, file_path):
    logger.info(f"Saving model to {file_path}")
    to_pickle(model, file_path)
    logger.info("Model saved successfully")


def load_model(file_path):
    logger.info(f"Loading model from {file_path}")
    model = joblib.load(file_path)
    logger.info("Model loaded successfully")
    return model
