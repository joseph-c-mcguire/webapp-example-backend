import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from app.models.model_manager import ModelManager
from sklearn.base import BaseEstimator


@pytest.fixture
def mock_config():
    with patch("app.models.model_manager.Config") as MockConfig:
        mock_config = MockConfig.return_value
        mock_config.MODEL_PATH = Path("/fake/path")
        yield mock_config


@pytest.fixture
def model_manager(mock_config):
    return ModelManager(mock_config.MODEL_PATH)


@patch("app.models.model_manager.joblib.load")
@patch("app.models.model_manager.Path.exists")
def test_load_model_success(mock_exists, mock_load, model_manager):
    mock_exists.return_value = True
    mock_model = MagicMock(spec=BaseEstimator)
    mock_load.return_value = mock_model

    model, error = model_manager.load_model("test_model")

    assert isinstance(model, BaseEstimator)
    assert error == ""
    mock_load.assert_called_once_with(Path("/fake/path/test_model.pkl"))


@patch("app.models.model_manager.Path.exists")
def test_load_model_file_not_found(mock_exists, model_manager):
    mock_exists.return_value = False

    model, error = model_manager.load_model("test_model")

    assert model is None
    assert error == "Model 'test_model' not found"


@patch("app.models.model_manager.joblib.load")
@patch("app.models.model_manager.Path.exists")
def test_load_model_exception(mock_exists, mock_load, model_manager):
    mock_exists.return_value = True
    mock_load.side_effect = Exception("Load error")

    model, error = model_manager.load_model("test_model")

    assert model is None
    assert error == "Load error"


@patch("app.models.model_manager.joblib.dump")
def test_save_model_success(mock_dump, model_manager):
    mock_model = MagicMock(spec=BaseEstimator)

    model_manager.save_model(mock_model, "test_model.pkl")

    mock_dump.assert_called_once_with(mock_model, Path("/fake/path/test_model.pkl"))
