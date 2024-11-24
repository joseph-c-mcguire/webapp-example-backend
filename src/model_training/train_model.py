import logging
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib
import json
from tqdm import tqdm
from sklearn.exceptions import NotFittedError

from src.data_utils import load_data, load_config, get_model
from src.ModelMonitor import ModelMonitor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_null_to_none(param_grid):
    for key, values in param_grid.items():
        param_grid[key] = [None if v == "null" else v for v in values]
    return param_grid


def validate_config(config):
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


def align_features(input_features, expected_features):
    """
    Align the input features with the expected features by adding missing columns with default values.

    Parameters:
    input_features (pd.DataFrame): Input features.
    expected_features (list): List of expected feature names.

    Returns:
    pd.DataFrame: Aligned features.
    """
    for feature in expected_features:
        if feature not in input_features.columns:
            input_features[feature] = 0  # or any default value
    return input_features[expected_features]


def train_model(config_path: str):
    # Load config
    logger.info("Loading configuration")
    config = load_config(config_path)
    if not config or not validate_config(config):
        logger.error("Failed to load or validate configuration. Exiting.")
        exit(1)

    # Load data
    logger.info("Loading data")
    data = load_data(config["data_path"]).drop(
        config.get("columns_to_drop", []), axis=1
    )
    if data.empty:
        logger.error("Failed to load data. Exiting.")
        exit(1)

    # Prepare features and target
    logger.info("Preparing features and target")
    X = data.drop(config["target_column"], axis=1)
    y = data[config["target_column"]]

    # Train-test split
    logger.info("Splitting data into train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, **config.get("train_test_split", {})
    )

    # Preprocess data using Pipeline
    logger.info("Setting up preprocessing pipeline")
    preprocessor = ColumnTransformer(
        transformers=[
            ("scale_columns", StandardScaler(), config.get("columns_to_scale", [])),
            ("encode_columns", OneHotEncoder(), config.get("columns_to_encode", [])),
        ]
    ).fit(X_train, y_train)

    # Save the preprocessor
    preprocessor_path = Path(config["model_directory"]) / "preprocessor.pkl"
    logger.info(f"Saving the preprocessor to {preprocessor_path}")
    joblib.dump(preprocessor, preprocessor_path)

    # Define models and their hyperparameter grids
    models = {
        model: get_model(
            meta_params["import_module"] + "." + meta_params["model_name"],
            meta_params["model_params"],
        )
        for model, meta_params in config["models"].items()
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
            param_grid = config["param_grids"].get(model_name, {})
            param_grid = convert_null_to_none(param_grid)
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
            try:
                grid_search.fit(preprocessor.fit_transform(X_train), y_train)
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
            with open(Path(config["model_directory"]) / "progress.json", "w") as f:
                json.dump(progress, f)

    # Save the results of each model
    results_path = Path(config["model_directory"]) / "model_results.json"
    logger.info(f"Saving model results to {results_path}")
    with open(results_path, "w") as f:
        json.dump(model_results, f)

    # Save the best model of each class
    model_directory = Path(config["model_directory"])
    model_directory.mkdir(parents=True, exist_ok=True)
    for model_name, best_model in best_models.items():
        model_path = model_directory / f"{model_name}.pkl"
        logger.info(f"Saving best model of {model_name} to {model_path}")
        joblib.dump(best_model, model_path)

    # Create a pipeline with the preprocessor and the best model
    best_model_name = max(model_results, key=model_results.get)
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", best_models[best_model_name])]
    )

    # Save the entire pipeline
    pipeline_path = Path(config["model_directory"]) / "best_model_pipeline.pkl"
    logger.info(f"Saving the entire pipeline to {pipeline_path}")
    joblib.dump(pipeline, pipeline_path)

    # Save the min and max values from the training data
    min_max_values = {"min": X_train.min().to_dict(), "max": X_train.max().to_dict()}
    min_max_values_path = Path(config["model_directory"]) / "min_max_values.pkl"
    logger.info(f"Saving the min and max values to {min_max_values_path}")
    joblib.dump(min_max_values, min_max_values_path)

    # Initialize and save the ModelMonitor
    logger.info("Initializing and saving the ModelMonitor")
    monitor = ModelMonitor(model=pipeline, X_train=preprocessor.transform(X_train))
    joblib.dump(monitor, Path(config["model_directory"]) / "model_monitor.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predictive Maintenance Model Training"
    )
    parser.add_argument(
        "--config",
        default="train_model.yaml",
        type=str,
        help="Path to the configuration file",
    )
    args = parser.parse_args()
    train_model(args.config)
