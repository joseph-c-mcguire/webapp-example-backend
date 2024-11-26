import pytest
from unittest.mock import patch, MagicMock
from app.services.training_service import TrainingService
from pandas import DataFrame


@pytest.fixture
def synthetic_data_path(tmp_path):
    data = DataFrame(
        {
            "UDI": [1, 2, 3, 4],
            "Product ID": ["M14860", "M14861", "M14862", "M14863"],
            "Type": ["M", "L", "M", "L"],
            "Air temperature [K]": [298.1, 298.2, 298.3, 298.4],
            "Process temperature [K]": [308.6, 308.7, 308.8, 308.9],
            "Rotational speed [rpm]": [1551, 1552, 1553, 1554],
            "Torque [Nm]": [42.8, 42.9, 43.0, 43.1],
            "Tool wear [min]": [0, 10, 20, 30],
            "Target": [0, 1, 0, 1],
            "Failure Type": [
                "No Failure",
                "Heat Dissipation Failure",
                "No Failure",
                "Power Failure",
            ],
        }
    )
    file_path = tmp_path / "synthetic_data.csv"
    data.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def mock_config(synthetic_data_path):
    return {
        "RAW_DATA_PATH": synthetic_data_path,
        "PROCESSED_DATA_PATH": synthetic_data_path,
        "MODEL_PATH": "path/to/models",
        "TARGET_COLUMN": "Target",
        "COLUMNS_TO_DROP": ["UDI", "Product ID"],
        "COLUMNS_TO_SCALE": [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ],
        "COLUMNS_TO_ENCODE": ["Type"],
        "TRAIN_TEST_SPLIT": {"test_size": 0.2, "random_state": 42},
        "MODEL_PARAMETERS": {
            "model1": {
                "import_module": "sklearn.ensemble",
                "model_name": "RandomForestClassifier",
                "model_params": {},
            }
        },
        "PARAM_GRIDS": {
            "model1": {"n_estimators": [10, 50, 100], "max_depth": [None, 10, 20]}
        },
    }


@pytest.fixture
def training_service(mock_config):
    with patch("app.config.Config", return_value=mock_config):
        return TrainingService("path/to/config.yaml")


@patch("app.services.training_service.load_data")
@patch("app.services.training_service.load_config")
@patch("app.services.training_service.joblib.dump")
@patch("app.services.training_service.get_model")
def test_train_model(
    mock_get_model, mock_joblib_dump, mock_load_config, mock_load_data, training_service
):
    # Mock data
    mock_load_data.return_value = DataFrame(
        {
            "UDI": [1, 2, 3, 4],
            "Product ID": ["M14860", "M14861", "M14862", "M14863"],
            "Type": ["M", "L", "M", "L"],
            "Air temperature [K]": [298.1, 298.2, 298.3, 298.4],
            "Process temperature [K]": [308.6, 308.7, 308.8, 308.9],
            "Rotational speed [rpm]": [1551, 1552, 1553, 1554],
            "Torque [Nm]": [42.8, 42.9, 43.0, 43.1],
            "Tool wear [min]": [0, 10, 20, 30],
            "Target": [0, 1, 0, 1],
            "Failure Type": [
                "No Failure",
                "Heat Dissipation Failure",
                "No Failure",
                "Power Failure",
            ],
        }
    )
    mock_load_config.return_value = training_service.config

    # Mock model
    mock_model = MagicMock()
    mock_model.fit.return_value = None
    mock_model.score.return_value = 0.9
    mock_get_model.return_value = mock_model

    result = training_service.train_model()

    assert result["status"] == "success"
    mock_joblib_dump.assert_called()
