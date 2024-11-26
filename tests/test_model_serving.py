import pytest
from unittest.mock import patch
from flask import Flask
from app.routes.model_serving import model_serving_bp


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(model_serving_bp)
    with app.test_client() as client:
        yield client


@patch("app.routes.model_serving.InferenceService")
def test_predict_probabilities_success(MockInferenceService, client):
    mock_service = MockInferenceService.return_value
    mock_service.predict_probabilities.return_value = {"probabilities": [0.1, 0.9]}

    response = client.post("/predict-probabilities")

    assert response.status_code == 200
    assert "probabilities" in response.json


@patch("app.routes.model_serving.InferenceService")
def test_predict_probabilities_no_data(MockInferenceService, client):
    mock_service = MockInferenceService.return_value
    mock_service.predict_probabilities.return_value = {"probabilities": [0.1, 0.9]}

    response = client.post("/predict-probabilities")

    assert response.status_code == 200
    assert "probabilities" in response.json


@patch("app.routes.model_serving.InferenceService")
def test_predict_probabilities_prediction_error(MockInferenceService, client):
    mock_service = MockInferenceService.return_value
    mock_service.predict_probabilities.return_value = {"error": "Model not found"}

    response = client.post(
        "/predict-probabilities",
        json={
            "model_name": "Unknown Model",
            "data": [
                {
                    "UDI": 1,
                    "Product ID": "M14860",
                    "Type": "M",
                    "Air temperature [K]": 298.1,
                    "Process temperature [K]": 308.6,
                    "Rotational speed [rpm]": 1551,
                    "Torque [Nm]": 42.8,
                    "Tool wear [min]": 0,
                    "Target": 0,
                    "Failure Type": "No Failure",
                }
            ],
        },
    )

    assert response.status_code == 400
    assert "error" in response.json
