import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from app import create_app  # Import the create_app function
from app.services.model_diagnostics import (
    get_roc_curve,
    get_feature_importance,  # Import the get_feature_importance function
)
from flask import Flask
from flask.testing import FlaskClient
from app.routes.model_diagnostics import model_diagnostics_bp


@pytest.fixture
def app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(model_diagnostics_bp)
    return app


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    return app.test_client()


def test_confusion_matrix_endpoint(client: FlaskClient, monkeypatch):
    monkeypatch.setattr(
        "app.routes.model_diagnostics.ModelManager.load_model",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(
        "app.routes.model_diagnostics.joblib.load", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "app.routes.model_diagnostics.get_confusion_matrix",
        lambda *args, **kwargs: ({}, 200),
    )

    response = client.post(
        "/confusion-matrix",
        json={"model_name": "Decision Tree", "class_label": "No Failure"},
    )
    assert response.status_code == 200


def test_roc_curve_endpoint(client: FlaskClient, monkeypatch):
    monkeypatch.setattr(
        "app.routes.model_diagnostics.ModelManager.load_model",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(
        "app.routes.model_diagnostics.joblib.load", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "app.routes.model_diagnostics.pd.read_csv",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6],
                "Failure Type": ["No Failure", "Failure", "No Failure"],
            }
        ),
    )
    monkeypatch.setattr(
        "app.routes.model_diagnostics.get_roc_curve", lambda *args, **kwargs: ({}, 200)
    )

    response = client.post(
        "/roc-curve", json={"model_name": "Decision Tree", "class_label": "No Failure"}
    )
    if response.status_code != 200:
        print(response.json)  # Log the error message for debugging
    assert response.status_code == 200


def test_roc_curve_endpoint_missing_json(client: FlaskClient):
    response = client.post("/roc-curve")
    assert response.status_code == 400
    assert response.json["error"] == "Invalid or missing JSON data"


def test_feature_importance_endpoint(client: FlaskClient, monkeypatch):
    monkeypatch.setattr(
        "app.routes.model_diagnostics.ModelManager.load_model",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(
        "app.routes.model_diagnostics.joblib.load",
        lambda *args, **kwargs: MagicMock(get_feature_names_out=lambda: []),
    )
    monkeypatch.setattr(
        "app.routes.model_diagnostics.get_feature_importance",
        lambda *args, **kwargs: ({}, 200),
    )

    response = client.get(
        "/feature-importance", query_string={"model_name": "Gradient Boosting"}
    )
    assert response.status_code == 200


def test_feature_importance_endpoint_missing_model(client: FlaskClient, monkeypatch):
    monkeypatch.setattr(
        "app.routes.model_diagnostics.ModelManager.load_model",
        lambda *args, **kwargs: (None, "Model not found"),
    )
    response = client.get(
        "/feature-importance", query_string={"model_name": "Decision Tree"}
    )
    assert response.status_code == 404
    assert response.json["error"] == "Model not found"


def test_feature_importance_endpoint_invalid_preprocessor(
    client: FlaskClient, monkeypatch
):
    monkeypatch.setattr(
        "app.routes.model_diagnostics.ModelManager.load_model",
        lambda *args, **kwargs: (MagicMock(), None),
    )
    monkeypatch.setattr(
        "app.routes.model_diagnostics.joblib.load",
        lambda *args, **kwargs: MagicMock(get_feature_names_out=lambda: None),
    )
    response = client.get(
        "/feature-importance", query_string={"model_name": "Decision Tree"}
    )
    assert response.status_code == 500
    assert "error" in response.json
    assert response.json["error"] == "Feature names list is empty or None."


@patch("app.services.model_diagnostics.pd.read_csv")
@patch(
    "app.services.model_diagnostics.Path.exists"
)  # Patch Path.exists instead of os.path.exists
def test_get_roc_curve(mock_exists, mock_read_csv, client, app):
    mock_exists.return_value = True
    mock_read_csv.return_value = pd.DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "Failure Type": [
                "No Failure",
                "Failure",
                "No Failure",
            ],  # Use string labels
        }
    )

    model = MagicMock()
    preprocessor = MagicMock()
    model.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    model.classes_ = ["No Failure", "Failure"]  # Use string classes
    preprocessor.transform.return_value = [[1, 4], [2, 5], [3, 6]]

    with app.app_context():
        response, status_code = get_roc_curve(
            model, preprocessor, "Decision Tree", "No Failure"
        )  # Pass class_label as string
        assert status_code == 200
        assert "fpr" in response.json
        assert "tpr" in response.json
        assert "roc_auc" in response.json


def test_get_feature_importance(client, app):
    model = MagicMock()
    model.feature_importances_ = [0.2, 0.8]
    feature_names = ["feature1", "feature2"]

    with app.app_context():
        response = get_feature_importance(model, feature_names)
        assert response.status_code == 200
        assert "feature_importance" in response.json
        assert response.json["feature_importance"] == {"feature1": 0.2, "feature2": 0.8}
