import os
import logging
import pandas as pd
import numpy as np  # Add this import
from flask import jsonify
from sklearn.metrics import confusion_matrix, roc_curve, auc

logger = logging.getLogger(__name__)


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
    test_file_path = os.path.join(os.path.dirname(__file__), "data", "test_data.csv")
    if not os.path.exists(test_file_path):
        logger.error(f"Test data file not found at {test_file_path}")
        return jsonify({"error": "Test data file not found"}), 404

    logger.info(f"Loading test data from {test_file_path}")
    df = pd.read_csv(test_file_path)
    df = df[
        df[class_label_column] == target_label
    ]  # Filter the data to only include the specified class label
    try:
        features = df.drop(class_label_column, axis=1)
        labels = df[class_label_column]
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


def get_roc_curve(model, preprocessor, model_name: str, class_label: int):
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
    test_file_path = os.path.join(os.path.dirname(__file__), "data", "test_data.csv")
    if not os.path.exists(test_file_path):
        logger.error(f"Test data file not found at {test_file_path}")
        return jsonify({"error": "Test data file not found"}), 404

    logger.info(f"Loading test data from {test_file_path}")
    df = pd.read_csv(test_file_path)
    try:
        features = df.drop(columns=[df.columns[class_label]])
        labels = df.iloc[:, class_label]
        features = preprocessor.transform(features)
        logger.info("Generating ROC curve data")
        probabilities = model.predict_proba(features)
        probabilities = np.array(probabilities)  # Convert probabilities to NumPy array
        class_index = list(model.classes_).index(class_label)
        fpr, tpr, _ = roc_curve(
            labels, probabilities[:, class_index], pos_label=class_label
        )
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
