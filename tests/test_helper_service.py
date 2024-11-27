import pytest
from flask import Flask
from app.routes.helper_service import helper_service_bp
from unittest.mock import patch


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(helper_service_bp)
    app.testing = True
    with app.test_client() as client:
        yield client


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
