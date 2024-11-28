import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from app import create_app  # Import the create_app function
from app.services.model_diagnostics import (
    get_roc_curve,
    get_feature_importance,  # Import the get_feature_importance function
)


@pytest.fixture
def app():
    app = create_app()  # Use the create_app function to create the Flask app
    return app


@pytest.fixture
def client(app):
    return app.test_client()


# @patch("app.services.model_diagnostics.pd.read_csv")
# @patch("app.services.model_diagnostics.Path.exists")
# def test_get_confusion_matrix(mock_exists, mock_read_csv, client, app):
#     mock_exists.return_value = True
#     mock_read_csv.return_value = pd.DataFrame(
#         {
#             "Type": ["L", "M", "H"],
#             "Air temperature [K]": [300, 305, 310],
#             "Process temperature [K]": [310, 315, 320],
#             "Rotational speed [rpm]": [1500, 1600, 1700],
#             "Torque [Nm]": [40, 42, 44],
#             "Tool wear [min]": [10, 20, 30],
#         }
#     )

#     model = MagicMock()
#     preprocessor = MagicMock()
#     model.predict.return_value = [0, 0]  # Adjusted to 2 samples
#     preprocessor.transform.return_value = [
#         [300, 310, 1500, 40, 10],
#         [310, 320, 1700, 44, 30],
#     ]  # Adjusted to 2 samples

#     with app.app_context():
#         response = client.post(
#             "/confusion-matrix",
#             json={"model_name": "Decision Tree"},
#         )
#         assert response.status_code == 200
#         # Add more assertions if needed to validate the response content


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
