import logging

from pandas import DataFrame

from app.models.model_manager import load_specific_model

logger = logging.getLogger(__name__)


def predict(model_name, features):
    """
    Predict equipment failure type using the specified model.

    Parameters:
    - model_name: str, name of the model to use for prediction
    - features: dict, input features for prediction

    Returns:
    - dict: prediction result
    """
    model, error = load_specific_model(model_name)
    if error:
        return {"error": error}

    try:
        features_df = DataFrame(features, index=[0])
        logger.info(f"Predicting with model: {model_name}")
        logger.info(f"Predicting with data: {features_df}")
        features_transformed = preprocessor.transform(features_df)
        logger.info(f"Transformed features: {features_transformed}")
        prediction = model.predict(features_transformed)
        return {"prediction": prediction[0]}
    except ValueError as e:
        logger.error(f"Error during prediction: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return {"error": str(e)}


def predict_probabilities(model_name, data):
    """
    Predict probabilities for equipment failure type using the specified model.

    Parameters:
    - model_name: str, name of the model to use for prediction
    - data: list of dict, input data for prediction

    Returns:
    - dict: prediction probabilities
    """
    model, error = load_specific_model(model_name)
    if error:
        return {"error": error}

    try:
        features_df = DataFrame(data)
        features_transformed = preprocessor.transform(features_df)
        logger.debug(f"Transformed features: {features_transformed}")

        if not hasattr(model, "predict_proba"):
            logger.error(f"Model {model_name} does not support probability prediction")
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
