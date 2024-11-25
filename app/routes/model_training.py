from flask import Blueprint, request, jsonify, current_app
import subprocess
import logging
from app.data_processing.split_data import split_data

bp = Blueprint("model_training", __name__)
logger = logging.getLogger(__name__)


@bp.route("/train", methods=["POST"])
def train_model():
    """
    Train the model by calling the train_model.py script.

    Returns:
    JSON response with the result of the training process.
    """
    train_val_file_path = DATA_PATH / "train_val_data.csv"
    test_file_path = DATA_PATH / "test_data.csv"
    logger.debug(f"Train/validation file path: {train_val_file_path}")
    logger.debug(f"Test file path: {test_file_path}")

    # Check if the train and test data files exist
    if not os.path.exists(train_val_file_path) or not os.path.exists(test_file_path):
        logger.info(
            "Train or test data file not found. Running split_data function to generate them."
        )
        try:
            split_data(
                data_file_path=DATA_PATH / "predictive_maintenance.csv",
                train_val_file_path=train_val_file_path,
                test_file_path=test_file_path,
            )
        except Exception as e:
            logger.error(f"An error occurred while splitting the data: {str(e)}")
            return (
                jsonify(
                    {
                        "error": "An error occurred while splitting the data",
                        "details": str(e),
                    }
                ),
                500,
            )

    # Proceed with training the model
    try:
        logger.info("Starting model training")
        result = subprocess.run(
            ["python", "train_model.py"], capture_output=True, text=True
        )
        logger.debug(f"Training result: {result.stdout}")
        if result.returncode == 0:
            logger.info("Model training completed successfully")
            return jsonify({"message": "Model training completed successfully"}), 200
        else:
            logger.error(f"Model training failed: {result.stderr}")
            logger.error(f"Model training stdout: {result.stdout}")
            return (
                jsonify({"error": "Model training failed", "details": result.stderr}),
                500,
            )
    except Exception as e:
        logger.error(f"An error occurred while training the model: {str(e)}")
        return (
            jsonify(
                {
                    "error": "An error occurred while training the model",
                    "details": str(e),
                },
            ),
            500,
        )


@bp.route("/retrain", methods=["POST"])
def retrain_model():
    """
    Retrain the models using the provided configuration.

    Request JSON format:
    {
        "data_path": "data/predictive_maintenance.csv",
        "model_directory": "models/",
        "columns_to_drop": ["Product ID"],
        "columns_to_scale": ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"],
        "columns_to_encode": ["Type"],
        "target_column": "Failure Type",
        "param_grids": {
            "Decision Tree": {"max_depth": [null, 10, 20, 30]},
            "Random Forest": {"n_estimators": [100, 200], "max_depth": [10, 20]},
            "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]},
        },
        "models": {
            "Decision Tree": {"import_module": "sklearn.tree", "model_name": "DecisionTreeClassifier", "model_params": {"random_state": 42}},
            "Random Forest": {"import_module": "sklearn.ensemble", "model_name": "RandomForestClassifier", "model_params": {"random_state": 42}},
            "Gradient Boosting": {"import_module": "sklearn.ensemble", "model_name": "GradientBoostingClassifier", "model_params": {"random_state": 42}},
        },
        "train_test_split": {"test_size": 0.2, "random_state": 42}
    }

    Returns:
    JSON response with the result of the retraining process.
    """
    config = request.get_json(force=True)
    config_path = os.path.join(BASE_DIR, "..", "..", "..", "retrain_config.json")
    logger.debug(f"Config path: {config_path}")

    # Save the configuration to a file
    with open(config_path, "w") as f:
        json.dump(config, f)

    # Proceed with retraining the model
    try:
        logger.info("Starting model retraining")
        result = subprocess.run(
            ["python", "train_model.py", "--config", config_path],
            capture_output=True,
            text=True,
        )
        logger.debug(f"Retraining result: {result.stdout}")
        if result.returncode == 0:
            logger.info("Model retraining completed successfully")
            return jsonify({"message": "Model retraining completed successfully"}), 200
        else:
            logger.error(f"Model retraining failed: {result.stderr}")
            logger.error(f"Model retraining stdout: {result.stdout}")
            return (
                jsonify({"error": "Model retraining failed", "details": result.stderr}),
                500,
            )
    except Exception as e:
        logger.error(f"An error occurred while retraining the model: {str(e)}")
        return (
            jsonify(
                {
                    "error": "An error occurred while retraining the model",
                    "details": str(e),
                },
            ),
            500,
        )
