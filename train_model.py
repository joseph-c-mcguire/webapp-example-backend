import logging
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib

from src.data_utils import (
    load_data, load_config, tune_model, get_model
)
from src.ModelMonitor import ModelMonitor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(config_path: str):
    # Load config
    logger.info("Loading configuration")
    config = load_config(config_path)
    if not config:
        logger.error("Failed to load configuration. Exiting.")
        exit(1)

    # Load data
    logger.info("Loading data")
    data = load_data(config["data_path"])
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
        X, y, **config.get("train_test_split", {}))

    # Preprocess data using Pipeline
    logger.info("Setting up preprocessing pipeline")
    preprocessor = ColumnTransformer(
        transformers=[
            ('drop_columns', 'drop', config.get("columns_to_drop", [])),
            ('scale_columns', StandardScaler(),
             config.get("columns_to_scale", [])),
            ('encode_columns', OneHotEncoder(),
             config.get("columns_to_encode", []))
        ]
    )

    # Define models and their hyperparameter grids
    models = {
        model: get_model(**meta_params) for model, meta_params in config["models"].items()
    }

    # Train and evaluate model
    # logger.info("Training and evaluating models")
    # results = train_and_evaluate_model(
    #     preprocessor.fit_transform(X_train), y_train, preprocessor.transform(X_test), y_test, models)
    # for model_name, result in results.items():
    #     logger.info(f"Model: {model_name}")
    #     logger.info(result['Classification Report'])
    #     logger.info(f"ROC AUC: {result['ROC AUC']}")

    # Tune model
    logger.info("Tuning models")
    best_model = tune_model(
        models, config["param_grids"], preprocessor.fit_transform(X_train), y_train)
    logger.info(f"Best model: {best_model}")

    # Create a pipeline with the preprocessor and the best model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', best_model)
    ])

    # Save the pipeline
    logger.info("Saving the pipeline")
    joblib.dump(pipeline, Path(config["model_directory"]) / 'best_model.pkl')

    # Save the min and max values from the training data
    min_max_values = {
        'min': X_train.min().to_dict(),
        'max': X_train.max().to_dict()
    }
    min_max_values_path = Path(
        config["model_directory"]) / 'min_max_values.pkl'
    logger.info(f"Saving the min and max values to {min_max_values_path}")
    joblib.dump(min_max_values, min_max_values_path)

    # Initialize and save the ModelMonitor
    logger.info("Initializing and saving the ModelMonitor")
    monitor = ModelMonitor(
        model=pipeline, X_train=preprocessor.transform(X_train))
    joblib.dump(monitor, Path(config["model_directory"]) / 'model_monitor.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Predictive Maintenance Model Training")
    parser.add_argument('--config', default="train_model.yaml", type=str,
                        help='Path to the configuration file')
    args = parser.parse_args()
    train_model(args.config)
