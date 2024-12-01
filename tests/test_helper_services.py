import pytest
import os
import json
import pandas as pd
from unittest.mock import patch, mock_open
from flask import Flask
from app.services.helper_service import (
    get_data,
    get_feature_names,
    get_model_results,
    get_training_progress,
    get_available_models,
    get_class_names,
)
from app.config import Config
from app.routes.helper_service import helper_service_bp

app = Flask(__name__)
app.register_blueprint(helper_service_bp)


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


@patch("app.services.helper_service.Config")
@patch("app.services.helper_service.os.path.exists")
@patch("app.services.helper_service.pd.read_csv")
def test_get_data(mock_read_csv, mock_exists, MockConfig, client):
    mock_exists.return_value = True
    mock_read_csv.return_value = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    MockConfig.return_value.TEST_DATA_PATH = "test_path"

    response = client.get("/data")
    assert response.status_code == 200
    assert response.json == [{"col1": 1, "col2": 3}, {"col1": 2, "col2": 4}]


@patch("app.services.helper_service.Config")
@patch("app.services.helper_service.os.path.exists")
def test_get_data_file_not_found(mock_exists, MockConfig, client):
    mock_exists.return_value = False
    MockConfig.return_value.TEST_DATA_PATH = "test_path"

    response = client.get("/data")
    assert response.status_code == 404
    assert response.json == {"error": "Data file not found"}


def test_get_feature_names(client):
    response = client.get("/feature-names")
    assert response.status_code == 200
    assert response.json == {"feature_names": ["feature1", "feature2", "feature3"]}


@patch("app.services.helper_service.os.path.exists")
@patch(
    "builtins.open", new_callable=mock_open, read_data='{"model1": {"accuracy": 0.9}}'
)
def test_get_model_results(mock_open, mock_exists, client):
    mock_exists.return_value = True

    response = client.get("/model-results")
    assert response.status_code == 200
    assert response.json == {"model_results": {"model1": {"accuracy": 0.9}}}


@patch("app.services.helper_service.os.path.exists")
def test_get_model_results_file_not_found(mock_exists, client):
    mock_exists.return_value = False

    response = client.get("/model-results")
    assert response.status_code == 404
    assert response.json == {"error": "Model results file not found"}


@patch("app.services.helper_service.os.path.exists")
@patch("builtins.open", new_callable=mock_open, read_data='{"progress": 50}')
def test_get_training_progress(mock_open, mock_exists, client):
    mock_exists.return_value = True

    response = client.get("/training-progress")
    assert response.status_code == 200
    assert response.json["progress"] == {"progress": 50}


@patch("app.services.helper_service.os.path.exists")
def test_get_training_progress_file_not_found(mock_exists, client):
    mock_exists.return_value = False

    response = client.get("/training-progress")
    assert response.status_code == 404
    assert response.get_json() == {"error": "Progress file not found"}


@patch("app.services.helper_service.Config")
def test_get_available_models(MockConfig, client):
    MockConfig.return_value.MODEL_PARAMETERS = {"model1": {}, "model2": {}}

    response = client.get("/available-models")
    assert response.status_code == 200
    assert response.json == {"available_models": ["model1", "model2"]}


@patch("app.services.helper_service.Config")
@patch("app.services.helper_service.os.path.exists")
@patch("app.services.helper_service.pd.read_csv")
def test_get_class_names(mock_read_csv, mock_exists, MockConfig, client):
    mock_exists.return_value = True
    mock_read_csv.return_value = pd.DataFrame({"target": ["class1", "class2"]})
    MockConfig.return_value.RAW_DATA_PATH = "test_path"
    MockConfig.return_value.TARGET_COLUMN = "target"

    response = client.get("/class-names")
    assert response.status_code == 200
    assert response.json == {"class_names": ["class1", "class2"]}


@patch("app.services.helper_service.Config")
@patch("app.services.helper_service.os.path.exists")
def test_get_class_names_file_not_found(mock_exists, MockConfig, client):
    mock_exists.return_value = False
    MockConfig.return_value.RAW_DATA_PATH = "test_path"

    response = client.get("/class-names")
    assert response.status_code == 404
    assert response.json == {"error": "Data file not found"}


def test_data_endpoint(client):
    with patch(
        "app.routes.helper_service.get_data", return_value="mocked data"
    ) as mock_get_data:
        response = client.get("/data")
        assert response.status_code == 200
        assert response.data == b"mocked data"
        mock_get_data.assert_called_once()


def test_feature_names_endpoint(client):
    with patch(
        "app.routes.helper_service.get_feature_names",
        return_value={"feature_names": ["feature1", "feature2", "feature3"]},
    ) as mock_get_feature_names:
        response = client.get("/feature-names")
        assert response.status_code == 200
        assert response.get_json() == {
            "feature_names": ["feature1", "feature2", "feature3"]
        }
        mock_get_feature_names.assert_called_once()


def test_model_results_endpoint(client):
    with patch(
        "app.routes.helper_service.get_model_results",
        return_value="mocked model results",
    ) as mock_get_model_results:
        response = client.get("/model-results")
        assert response.status_code == 200
        assert response.data == b"mocked model results"
        mock_get_model_results.assert_called_once()


def test_training_progress_endpoint(client):
    with patch(
        "app.routes.helper_service.get_training_progress",
        return_value="mocked training progress",
    ) as mock_get_training_progress:
        response = client.get("/training-progress")
        assert response.status_code == 200
        assert response.data == b"mocked training progress"
        mock_get_training_progress.assert_called_once()


def test_available_models_endpoint(client):
    with patch(
        "app.routes.helper_service.get_available_models",
        return_value="mocked available models",
    ) as mock_get_available_models:
        response = client.get("/available-models")
        assert response.status_code == 200
        assert response.data == b"mocked available models"
        mock_get_available_models.assert_called_once()


def test_class_names_endpoint(client):
    with patch(
        "app.routes.helper_service.get_class_names", return_value="mocked class names"
    ) as mock_get_class_names:
        response = client.get("/class-names")
        assert response.status_code == 200
        assert response.data == b"mocked class names"
        mock_get_class_names.assert_called_once()
