import logging
import joblib

import json
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm import tqdm
from pandas import DataFrame

from app.utils.data_utils import load_data, load_config
from app.utils.model_utils import get_model
from app.models.model_manager import ModelManager  # Import ModelManager
from typing import Dict, Any
from app.config import Config  # Ensure Config is imported

# Set up logging
logger = logging.getLogger(__name__)


class TrainingService:
    """
    Service for training machine learning models based on a configuration file.

    Attributes
    ----------
    config_path : str
        Path to the configuration file.
    config : Config
        Loaded configuration settings.
    model_manager : ModelManager
        Manages model saving and retrieval.
    raw_data_path : Path
        Path to the raw dataset.
    processed_data_path : Path
        Path where processed data is stored.
    """

    def __init__(self, config_path: str):
        """
        Initialize the TrainingService with configuration settings.

        Parameters
        ----------
        config_path : str
            Path to the configuration file.
        """
        self.config = Config(config_path)  # Pass config_path to Config
        self.model_manager = ModelManager(self.config.MODEL_PATH)
        self.raw_data_path = self.config.RAW_DATA_PATH
        self.processed_data_path = self.config.PROCESSED_DATA_PATH

    def load_and_validate_config(self) -> Dict[str, Any]:
        """
        Load and validate the training configuration.

        Returns
        -------
        Dict[str, Any]
            The validated configuration dictionary.

        Raises
        ------
        SystemExit
            If the configuration fails to load or validate.
        """
        logger.info("Loading configuration")
        config = load_config(self.config_path)
        if not config or not self.validate_config(config):
            logger.error("Failed to load or validate configuration. Exiting.")
            exit(1)
        return config

    @staticmethod
    def convert_null_to_none(param_grid: Dict[str, Any]) -> Dict[str, Any]:
        """
        Replace "null" strings with None in the parameter grid.

        Parameters
        ----------
        param_grid : Dict[str, Any]
            Hyperparameter grid with possible "null" values.

        Returns
        -------
        Dict[str, Any]
            Parameter grid with "null" values converted to None.
        """
        for key, values in param_grid.items():
            param_grid[key] = [None if v == "null" else v for v in values]
        return param_grid

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        Validate the presence of required keys in the configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary to validate.

        Returns
        -------
        bool
            True if all required keys are present, False otherwise.
        """
        required_keys = [
            "data_path",
            "target_column",
            "model_directory",
            "models",
            "param_grids",
        ]
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required config key: {key}")
                return False
        return True

    @staticmethod
    def align_features(input_features: DataFrame, expected_features: list) -> DataFrame:
        """
        Ensure input features match the expected feature set by adding missing columns.

        Parameters
        ----------
        input_features : DataFrame
            DataFrame containing input features.
        expected_features : list
            List of feature names expected by the model.

        Returns
        -------
        DataFrame
            Aligned DataFrame with all expected features.
        """
        for feature in expected_features:
            if feature not in input_features.columns:
                input_features[feature] = 0  # or any default value
        return input_features[expected_features]

    def train_model(self) -> Dict[str, Any]:
        """
        Train and evaluate machine learning models based on the configuration.

        Returns
        -------
        Dict[str, Any]
            Status dictionary indicating success or failure.
        """
        # Load raw data
        logger.info("Loading raw data")
        raw_data = load_data(self.raw_data_path).drop(
            self.config.COLUMNS_TO_DROP, axis=1
        )
        # Save processed data
        raw_data.to_csv(self.processed_data_path / "train_val_data.csv", index=False)
        # Load processed data
        logger.info("Loading processed data")
        data = load_data(self.processed_data_path / "train_val_data.csv")
        if data.empty:
            logger.error("Failed to load data. Exiting.")
            exit(1)

        # Prepare features and target
        logger.info("Preparing features and target")
        X = data.drop(
            self.config.TARGET_COLUMN, axis=1
        )  # Updated from TAR to TARGET_COLUMN
        y = data[self.config.TARGET_COLUMN]  # Ensure consistent usage

        # Train-test split
        logger.info("Splitting data into train and test sets")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            **self.config.TRAIN_TEST_SPLIT,  # Updated from self.config.get("train_test_split", {})
        )

        # Preprocess data using Pipeline
        logger.info("Setting up preprocessing pipeline")
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "scale_columns",
                    StandardScaler(),
                    self.config.COLUMNS_TO_SCALE,
                ),
                (
                    "encode_columns",
                    OneHotEncoder(),
                    self.config.COLUMNS_TO_ENCODE,
                ),
            ]
        ).fit(X_train, y_train)

        # Save the preprocessor in MODEL_PATH
        preprocessor_path = self.config.MODEL_PATH / "preprocessor.pkl"
        logger.info(f"Saving the preprocessor to {preprocessor_path}")
        joblib.dump(preprocessor, preprocessor_path)

        # Define models and their hyperparameter grids
        models = {
            model: get_model(
                meta_params["import_module"] + "." + meta_params["model_name"],
                meta_params["model_params"],
            )
            for model, meta_params in self.config.MODEL_PARAMETERS.items()
        }

        # Train and evaluate model
        logger.info("Training and evaluating models")
        model_results = {}
        best_models = {}
        progress = {}
        total_models = len(models)
        with tqdm(total=total_models, desc="Training Models") as pbar:
            for model_name, model in models.items():
                logger.info(f"Training model: {model_name}")
                param_grid = self.config.PARAM_GRIDS.get(model_name, {})
                param_grid = self.convert_null_to_none(param_grid)
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
                try:
                    grid_search.fit(preprocessor.transform(X_train), y_train)
                    best_model = grid_search.best_estimator_
                    score = best_model.score(preprocessor.transform(X_test), y_test)
                    model_results[model_name] = score
                    best_models[model_name] = best_model
                    logger.info(f"Model: {model_name}, Best Score: {score}")
                except ValueError as e:
                    logger.warning(f"Skipping model {model_name} due to error: {e}")
                except NotFittedError as e:
                    logger.error(f"Model {model_name} could not be fitted: {e}")
                progress[model_name] = "completed"
                pbar.update(1)
                # Save progress to a file
                with open(self.config.MODEL_PATH / "progress.json", "w") as f:
                    json.dump(progress, f)

        # Save the results of each model
        results_path = (
            self.config.MODEL_DIRECTORY / "model_results.json"
        )  # Updated from self.config["model_directory"]

        logger.info(f"Saving model results to {results_path}")
        with open(results_path, "w") as f:
            json.dump(model_results, f)

        # Save the best model of each class using ModelManager
        for model_name, best_model in best_models.items():
            logger.info(f"Saving best model of {model_name}")
            self.model_manager.save_model(best_model, model_name)

        return {"status": "success"}
