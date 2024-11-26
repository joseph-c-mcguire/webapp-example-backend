import pytest
from unittest.mock import MagicMock, patch
from pandas import DataFrame
from app.services.inference_service import InferenceService
from sklearn.preprocessing import StandardScaler
from pathlib import Path


@pytest.fixture
def inference_service():
    preprocessor = StandardScaler()
    model_path = Path("models/")
    return InferenceService(preprocessor, model_path)


def test_predict_success(inference_service):
    model_mock = MagicMock()
    model_mock.predict.return_value = [1]
    inference_service.model_manager.load_model = MagicMock(
        return_value=(model_mock, None)
    )

    # Mock the preprocessor's transform method
    inference_service.preprocessor.transform = MagicMock(return_value=[[1.0, 2.0]])

    features = {"feature1": 1.0, "feature2": 2.0}
    result = inference_service.predict("test_model", features)

    assert "prediction" in result
    assert result["prediction"] == 1


def test_predict_model_load_error(inference_service):
    inference_service.model_manager.load_model = MagicMock(
        return_value=(None, "Model not found")
    )

    features = {"feature1": 1.0, "feature2": 2.0}
    result = inference_service.predict("test_model", features)

    assert "error" in result
    assert result["error"] == "Model not found"


def test_predict_value_error(inference_service):
    model_mock = MagicMock()
    model_mock.predict.side_effect = ValueError("Invalid input")
    inference_service.model_manager.load_model = MagicMock(
        return_value=(model_mock, None)
    )

    # Mock the preprocessor's transform method
    inference_service.preprocessor.transform = MagicMock(return_value=[[1.0, 2.0]])

    features = {"feature1": 1.0, "feature2": 2.0}
    result = inference_service.predict("test_model", features)

    assert "error" in result
    assert result["error"] == "Invalid input"


def test_predict_probabilities_success(inference_service):
    model_mock = MagicMock()
    model_mock.predict_proba.return_value = MagicMock()
    model_mock.predict_proba.return_value.tolist.return_value = [[0.1, 0.9]]
    inference_service.model_manager.load_model = MagicMock(
        return_value=(model_mock, None)
    )

    # Mock the preprocessor's transform method
    inference_service.preprocessor.transform = MagicMock(return_value=[[1.0, 2.0]])

    data = [{"feature1": 1.0, "feature2": 2.0}]
    result = inference_service.predict_probabilities("test_model", data)

    assert "probabilities" in result
    assert result["probabilities"] == [[0.1, 0.9]]


def test_predict_probabilities_model_load_error(inference_service):
    inference_service.model_manager.load_model = MagicMock(
        return_value=(None, "Model not found")
    )

    data = [{"feature1": 1.0, "feature2": 2.0}]
    result = inference_service.predict_probabilities("test_model", data)

    assert "error" in result
    assert result["error"] == "Model not found"


def test_predict_probabilities_invalid_data_format(inference_service):
    model_mock = MagicMock()
    inference_service.model_manager.load_model = MagicMock(
        return_value=(model_mock, None)
    )

    data = "invalid data format"
    result = inference_service.predict_probabilities("test_model", data)

    assert "error" in result
    assert result["error"] == "Invalid data format. Expected a list of dictionaries."


def test_predict_probabilities_no_predict_proba(inference_service):
    model_mock = MagicMock()
    del model_mock.predict_proba
    inference_service.model_manager.load_model = MagicMock(
        return_value=(model_mock, None)
    )

    # Mock the preprocessor's transform method
    inference_service.preprocessor.transform = MagicMock(return_value=[[1.0, 2.0]])

    data = [{"feature1": 1.0, "feature2": 2.0}]
    result = inference_service.predict_probabilities("test_model", data)

    assert "error" in result
    assert result["error"] == "Model test_model does not support probability prediction"
