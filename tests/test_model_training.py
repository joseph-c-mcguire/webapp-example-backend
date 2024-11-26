import pytest
from unittest.mock import patch, MagicMock
from flask import Flask
from app.routes.model_training import model_training_bp


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(model_training_bp)
    with app.test_client() as client:
        yield client


@patch("app.routes.model_training.split_data")
@patch("app.routes.model_training.TrainingService")
@patch("app.routes.model_training.Config")
def test_train_model_success(
    mock_config, mock_training_service, mock_split_data, client
):
    mock_config_instance = mock_config.return_value
    mock_config_instance.RAW_DATA_PATH = "/fake/path"
    mock_config_instance.PROCESSED_DATA_PATH = "/fake/processed/path"
    mock_training_service_instance = mock_training_service.return_value
    mock_training_service_instance.train_model.return_value = {"status": "success"}

    response = client.post("/train")
    assert response.status_code == 200
    assert "Model training completed successfully" in response.get_json()["message"]


@patch("app.routes.model_training.split_data")
@patch("app.routes.model_training.TrainingService")
@patch("app.routes.model_training.Config")
def test_train_model_split_data_error(
    mock_config, mock_training_service, mock_split_data, client
):
    mock_config_instance = mock_config.return_value
    mock_config_instance.RAW_DATA_PATH = "/fake/path"
    mock_config_instance.PROCESSED_DATA_PATH = "/fake/processed/path"
    mock_split_data.side_effect = Exception("Split data error")

    response = client.post("/train")
    assert response.status_code == 500
    assert "An error occurred while splitting the data" in response.get_json()["error"]


@patch("app.routes.model_training.split_data")
@patch("app.routes.model_training.TrainingService")
@patch("app.routes.model_training.Config")
def test_train_model_training_error(
    mock_config, mock_training_service, mock_split_data, client
):
    mock_config_instance = mock_config.return_value
    mock_config_instance.RAW_DATA_PATH = "/fake/path"
    mock_config_instance.PROCESSED_DATA_PATH = "/fake/processed/path"
    mock_training_service_instance = mock_training_service.return_value
    mock_training_service_instance.train_model.side_effect = Exception("Training error")

    response = client.post("/train")
    assert response.status_code == 500
    assert "An error occurred while training the model" in response.get_json()["error"]
