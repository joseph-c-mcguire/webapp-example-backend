import logging
from flask import Blueprint
from app.services.helper_service import (
    get_data,
    get_feature_names,
    get_model_results,
    get_training_progress,
    get_available_models,
    get_class_names,
)

helper_service_bp = Blueprint("helper_service", __name__)
logger = logging.getLogger(__name__)


@helper_service_bp.route("/data", methods=["GET"])
def data_endpoint():
    """
    Endpoint to retrieve data from the predictive maintenance CSV file.

    Returns
    -------
    flask.Response
        JSON response with the data or an error message.
    """
    return get_data()


@helper_service_bp.route("/feature-names", methods=["GET"])
def feature_names_endpoint():
    """
    Get the feature names used during training.

    Returns
    -------
    flask.Response
        JSON response with the feature names.
    """
    return get_feature_names()


@helper_service_bp.route("/model-results", methods=["GET"])
def model_results_endpoint():
    """
    Get the results of each model or a specific model.

    Query parameter:
    - model_name: str, optional, name of the model to query

    Returns
    -------
    flask.Response
        JSON response with the results of each model or the specified model.
    """
    return get_model_results()


@helper_service_bp.route("/training-progress", methods=["GET"])
def training_progress_endpoint():
    """
    Get the training progress.

    Returns
    -------
    flask.Response
        JSON response with the training progress.
    """
    return get_training_progress()


@helper_service_bp.route("/available-models", methods=["GET"])
def available_models_endpoint():
    """
    Get the list of available models from the train_model.yaml configuration.

    Returns
    -------
    flask.Response
        JSON response with the list of available models.
    """
    return get_available_models()


@helper_service_bp.route("/class-names", methods=["GET"])
def class_names_endpoint():
    """
    Get the class names from the Failure Type column in the data.

    Returns
    -------
    flask.Response
        JSON response with the list of class names.
    """
    return get_class_names()
