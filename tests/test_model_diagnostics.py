import pytest
import pandas as pd
import numpy as np  # Add this import
from unittest.mock import MagicMock, patch
from flask import Flask
from app.services.model_diagnostics import (
    get_confusion_matrix,
    get_roc_curve,
    get_feature_importance,
)


@pytest.fixture
def app():
    app = Flask(__name__)
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


@patch("app.services.model_diagnostics.pd.read_csv")
@patch(
    "app.services.model_diagnostics.Path.exists"
)  # Patch Path.exists instead of os.path.exists
def test_get_confusion_matrix(mock_exists, mock_read_csv, client, app):
    mock_exists.return_value = True
    mock_read_csv.return_value = pd.DataFrame(
        {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "class_label": ["A", "B", "A"]}
    )

    model = MagicMock()
    preprocessor = MagicMock()
    model.predict.return_value = ["A", "A"]  # Adjusted to 2 samples
    preprocessor.transform.return_value = [[1, 4], [3, 6]]  # Adjusted to 2 samples

    with app.app_context():
        response, status_code = get_confusion_matrix(
            model, preprocessor, "model_name", "class_label", "A"  # Pass target_label
        )
        assert status_code == 200
        assert "confusion_matrix" in response.json


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
