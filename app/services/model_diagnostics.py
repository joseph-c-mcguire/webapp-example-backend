import logging
import pandas as pd
import numpy as np  # Add this import
from flask import jsonify
from sklearn.metrics import confusion_matrix, roc_curve, auc
from app.config import Config
from pathlib import Path  # Import Path

logger = logging.getLogger(__name__)

config = Config()  # Add this line to instantiate Config


def get_confusion_matrix(
    model, preprocessor, model_name: str, class_label_column: str, target_label
):
    """
    Get the confusion matrix for the specified model using test data from test_data.csv.

    Parameters
    ----------
    model : BaseEstimator
        The trained model.
    preprocessor : BaseEstimator
        The preprocessor for transforming input features.
    model_name : str
        Name of the model to use for prediction.
    class_label_column : str
        The class label column to filter the data.
    target_label : str
        The target class label value to filter the data.

    Returns
    -------
    flask.Response
        JSON response with the confusion matrix or an error message.
    """
    test_data_path = config.TEST_DATA_PATH  # Use Config to get test data path
    if not test_data_path.exists():
        logger.error(f"Test data file not found at {test_data_path}")
        return jsonify({"error": f"Test data file not found at {test_data_path}"}), 404

    logger.info(f"Loading test data from {test_data_path}")
    test_data = pd.read_csv(test_data_path)
    df = test_data
    try:
        features = df.drop(config.TARGET_COLUMN, axis=1)
        labels = df[config.TARGET_COLUMN]
        features = preprocessor.transform(features)
        logger.info("Calculating confusion matrix")
        predictions = model.predict(features)
        cm = confusion_matrix(labels, predictions)
        logger.debug(f"Confusion matrix: {cm}")
        return (
            jsonify({"confusion_matrix": cm.tolist()}),
            200,
        )  # Ensure consistent return type
    except Exception as e:
        logger.error(f"Error getting confusion matrix: {e}")
        return jsonify({"error": str(e)}), 400


def get_roc_curve(model, preprocessor, model_name: str, class_label: str):
    """
    Generate the ROC curve data for the specified model using test data from test_data.csv.

    Parameters
    ----------
    model : BaseEstimator
        The trained model.
    preprocessor : BaseEstimator
        The preprocessor for transforming input features.
    model_name : str
        Name of the model to use for prediction.
    class_label : str
        The class label to filter the data.

    Returns
    -------
    flask.Response
        JSON response with the false positive rate, true positive rate, and AUC.
    """
    test_data_path = config.TEST_DATA_PATH  # Use Config to get test data path
    if not test_data_path.exists():
        logger.error(f"Test data file not found at {test_data_path}")
        return (
            jsonify(
                {
                    "error": "Test data file not found. Please ensure 'test_data.csv' exists in the 'data/processed' directory."
                }
            ),
            404,
        )

    logger.info(f"Loading test data from {test_data_path}")
    test_data = pd.read_csv(test_data_path)
    df = test_data
    try:
        features = df.drop(columns=[config.TARGET_COLUMN])
        labels = df[config.TARGET_COLUMN].apply(lambda x: 1 if x == class_label else 0)
        features = preprocessor.transform(features)
        logger.info("Generating ROC curve data")
        probabilities = model.predict_proba(features)
        probabilities = np.array(probabilities)
        fpr, tpr, _ = roc_curve(
            labels, probabilities[:, 1]
        )  # Assuming class_label is the positive class
        roc_auc = auc(fpr, tpr)
        logger.debug(f"ROC AUC: {roc_auc}")
        return (
            jsonify({"fpr": fpr.tolist(), "tpr": tpr.tolist(), "roc_auc": roc_auc}),
            200,
        )
    except KeyError as e:
        logger.error(f"Class label {class_label} not found in model classes: {e}")
        return (
            jsonify({"error": f"Class label {class_label} not found in model classes"}),
            400,
        )
    except Exception as e:
        logger.error(f"Error getting ROC curve: {e}")
        return jsonify({"error": str(e)}), 400


def get_feature_importance(model, feature_names: list):
    """
    Get the feature importance of the specified model.

    Parameters
    ----------
    model : BaseEstimator
        The trained model.
    feature_names : list
        List of feature names.

    Returns
    -------
    flask.Response
        JSON response with the feature importance or an error message.
    """
    try:
        # Assuming the model has a feature_importances_ attribute
        feature_importances = model.feature_importances_
        feature_importance_dict = dict(zip(feature_names, feature_importances))
        logger.debug(f"Feature importances: {feature_importance_dict}")
        return jsonify({"feature_importance": feature_importance_dict})
    except AttributeError as e:
        logger.error(f"Model does not have feature_importances_ attribute: {str(e)}")
        return (
            jsonify(
                {
                    "error": "Model does not have feature_importances_ attribute",
                    "details": str(e),
                }
            ),
            500,
        )
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        return jsonify({"error": str(e), "details": str(e)}), 500
