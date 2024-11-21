import logging
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from pandas import DataFrame
import joblib
import pandas as pd
import subprocess
from sklearn.metrics import confusion_matrix, roc_curve, auc
import json

from src.data_utils import load_model, load_config
from split_data import split_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model and ModelMonitor
model_path = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "models", "best_model.pkl"))
monitor_path = os.getenv(
    "MONITOR_PATH", os.path.join(BASE_DIR, "models", "model_monitor.pkl")
)
config_path = os.getenv("CONFIG_PATH", os.path.join(BASE_DIR, "train_model.yaml"))
logger.info(f"Loading the trained model from {model_path}")
pipeline = load_model(model_path)
logger.info(f"Loading the ModelMonitor from {monitor_path}")
monitor = joblib.load(monitor_path)
logger.info(f"Loading the Configuration File from {config_path}")
config = load_config(config_path)

# Load the min and max values from the training data
min_max_values_path = os.getenv(
    "MIN_MAX_VALUES_PATH", os.path.join(BASE_DIR, "models", "min_max_values.pkl")
)
logger.info(f"Loading the min and max values from {min_max_values_path}")
min_max_values = joblib.load(min_max_values_path)

# Load the preprocessor
preprocessor_path = os.getenv(
    "PREPROCESSOR_PATH", os.path.join(BASE_DIR, "models", "preprocessor.pkl")
)
logger.info(f"Loading the preprocessor from {preprocessor_path}")
preprocessor = joblib.load(preprocessor_path)

# Get the feature names used during training
feature_names = []
for transformer in preprocessor.transformers:
    if transformer[0] != "drop_columns":
        feature_names.extend(transformer[2])


def load_specific_model(model_name):
    model_path = os.path.join(BASE_DIR, "models", f"{model_name}.pkl")
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        return None, f"Model '{model_name}' not found"
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    return model, None


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict equipment failure type using the specified model.

    Request JSON format:
    {
        "model_name": "LogisticRegression",
        "features": {
            "Type": "M",
            "Air temperature [K]": 300,
            "Process temperature [K]": 310,
            "Rotational speed [rpm]": 1500,
            "Torque [Nm]": 40,
            "Tool wear [min]": 10
        }
    }

    Returns:
    JSON response with prediction result.
    """
    if not pipeline:
        return jsonify({"error": "Model not loaded"}), 500
    data = request.get_json(force=True)
    logger.info(f"Received data: {data}")
    model_name = data.get("model_name", "Logistic Regression")
    logger.info(f"Model name: {model_name}")
    model, error = load_specific_model(model_name)
    if error:
        return jsonify({"error": error}), 404

    try:
        features = DataFrame(data["features"], index=[0])
        logger.info(f"Predicting with model: {model_name}")
        logger.info(f"Predicting with data: {features}")
        features = preprocessor.transform(features)
        logger.info(f"Transformed features: {features}")
        prediction = model.predict(features)
        return jsonify({"prediction": prediction[0]})
    except ValueError as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400


@app.route("/predict-probabilities", methods=["POST"])
def predict_probabilities():
    """
    Predict probabilities for equipment failure type using the specified model.

    Request JSON format:
    {
        "model_name": "LogisticRegression",
        "data": [
            {
                "Type": "M",
                "Air temperature [K]": 300,
                "Process temperature [K]": 310,
                "Rotational speed [rpm]": 1500,
                "Torque [Nm]": 40,
                "Tool wear [min]": 10
            }
        ]
    }

    Returns:
    JSON response with prediction probabilities.
    """
    if not pipeline:
        return jsonify({"error": "Model not loaded"}), 500
    data = request.get_json(force=True)
    model_name = data.get("model_name", "Logistic Regression")
    logger.debug(f"Model name: {model_name}")
    model, error = load_specific_model(model_name)
    if error:
        return jsonify({"error": error}), 404

    try:
        features = DataFrame(data["data"])
        features = preprocessor.transform(features)
        logger.debug(f"Transformed features: {features}")
        probabilities = model.predict_proba(features)  # Multi-class probabilities
        return jsonify({"probabilities": probabilities.tolist()})
    except ValueError as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400


@app.route("/data", methods=["GET"])
def get_data():
    data_file_path = os.path.join(
        os.path.dirname(__file__), "data", "predictive_maintenance.csv"
    )
    logger.debug(f"Data file path: {data_file_path}")
    if not os.path.exists(data_file_path):
        logger.error(f"Data file not found at {data_file_path}")
        return jsonify({"error": "Data file not found"}), 404
    df = pd.read_csv(data_file_path)
    data = df.to_dict(orient="records")
    return jsonify(data)


@app.route("/train", methods=["POST"])
def train_model():
    """
    Train the model by calling the train_model.py script.

    Returns:
    JSON response with the result of the training process.
    """
    train_val_file_path = os.path.join(
        os.path.dirname(__file__), "data", "train_val_data.csv"
    )
    test_file_path = os.path.join(os.path.dirname(__file__), "data", "test_data.csv")
    logger.debug(f"Train/validation file path: {train_val_file_path}")
    logger.debug(f"Test file path: {test_file_path}")

    # Check if the train and test data files exist
    if not os.path.exists(train_val_file_path) or not os.path.exists(test_file_path):
        logger.info(
            "Train or test data file not found. Running split_data function to generate them."
        )
        try:
            split_data(
                data_file_path=os.path.join(
                    os.path.dirname(__file__), "data", "predictive_maintenance.csv"
                ),
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
                }
            ),
            500,
        )


@app.route("/confusion-matrix", methods=["POST"])
def get_confusion_matrix():
    """
    Get the confusion matrix for the specified model using test data from test_data.csv.

    Request JSON format:
    {
        "model_name": "LogisticRegression",
        "class_label": "class_name"
    }

    Returns:
    JSON response with the confusion matrix.
    """
    if not pipeline:
        return jsonify({"error": "Model not loaded"}), 500
    data = request.get_json(force=True)
    model_name = data.get("model_name", "Logistic Regression")
    class_label = data.get("class_label", "No Failure")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Class label: {class_label}")
    model, error = load_specific_model(model_name)
    if error:
        return jsonify({"error": error}), 404

    test_file_path = os.path.join(os.path.dirname(__file__), "data", "test_data.csv")
    if not os.path.exists(test_file_path):
        logger.error(f"Test data file not found at {test_file_path}")
        return jsonify({"error": "Test data file not found"}), 404

    logger.info(f"Loading test data from {test_file_path}")
    df = pd.read_csv(test_file_path)
    ## Filter the data to only include the specified class label
    df = df[df[config["target_column"]] == class_label]
    try:
        features = df.drop(config["target_column"], axis=1)
        labels = df[config["target_column"]]
        features = preprocessor.transform(features)
        logger.info("Calculating confusion matrix")
        predictions = model.predict(features)
        cm = confusion_matrix(labels, predictions, labels=[class_label])
        logger.debug(f"Confusion matrix: {cm}")
        return jsonify({"confusion_matrix": cm.tolist()})
    except Exception as e:
        logger.error(f"Error getting confusion matrix: {e}")
        return jsonify({"error": str(e)}), 400


@app.route("/roc-curve", methods=["POST"])
def get_roc_curve():
    """
    Generate the ROC curve data for the specified model using test data from test_data.csv.

    Request JSON format:
    {
        "model_name": "LogisticRegression",
        "class_label": "class_name"
    }

    Returns:
    JSON response with the false positive rate, true positive rate, and AUC.
    """
    if not pipeline:
        return jsonify({"error": "Model not loaded"}), 500
    data = request.get_json(force=True)
    logger.info(f"Received data for ROC curve: {data}")
    model_name = data.get("model_name", "Logistic Regression")
    class_label = data.get("class_label", "No Failure")

    if not model_name or not class_label:
        logger.error("Missing model_name or class_label in request data")
        return jsonify({"error": "Missing model_name or class_label"}), 400

    logger.debug(f"Model name: {model_name}")
    logger.debug(f"Class label: {class_label}")
    model, error = load_specific_model(model_name)
    if error:
        return jsonify({"error": error}), 404

    test_file_path = os.path.join(os.path.dirname(__file__), "data", "test_data.csv")
    if not os.path.exists(test_file_path):
        logger.error(f"Test data file not found at {test_file_path}")
        return jsonify({"error": "Test data file not found"}), 404

    logger.info(f"Loading test data from {test_file_path}")
    df = pd.read_csv(test_file_path)
    try:
        features = df.drop(config["target_column"], axis=1)
        labels = df[config["target_column"]]
        logger.debug(f"Features shape: {features.shape}")
        logger.debug(f"Labels shape: {labels.shape}")
        features = pipeline.named_steps["preprocessor"].transform(features)
        logger.info("Generating ROC curve data")
        probabilities = model.predict_proba(features)
        class_index = list(model.classes_).index(class_label)
        fpr, tpr, _ = roc_curve(
            labels, probabilities[:, class_index], pos_label=class_label
        )
        roc_auc = auc(fpr, tpr)
        logger.debug(f"ROC AUC: {roc_auc}")
        return jsonify({"fpr": fpr.tolist(), "tpr": tpr.tolist(), "roc_auc": roc_auc})
    except KeyError as e:
        logger.error(f"Class label {class_label} not found in model classes: {e}")
        return (
            jsonify({"error": f"Class label {class_label} not found in model classes"}),
            400,
        )
    except Exception as e:
        logger.error(f"Error getting ROC curve: {e}")
        return jsonify({"error": str(e)}), 400


@app.route("/feature-importance", methods=["GET"])
def get_feature_importance():
    """
    Get the feature importance of the specified model.

    Query parameter:
    - model_name: str, optional, name of the model to query

    Returns:
    JSON response with the feature importance.
    """
    if not pipeline:
        return jsonify({"error": "Model not loaded"}), 500
    model_name = request.args.get("model_name", "Gradient Boosting")
    logger.debug(f"Model name: {model_name}")
    model, error = load_specific_model(model_name)
    if error:
        return jsonify({"error": error}), 404

    logger.info("Getting feature importance")
    try:
        # Assuming the model has a feature_importances_ attribute
        feature_importances = model.feature_importances_
        feature_importance_dict = dict(zip(feature_names, feature_importances))
        logger.debug(f"Feature importances: {feature_importance_dict}")
        return jsonify({"feature_importance": feature_importance_dict})
    except AttributeError as e:
        logger.error(f"Model does not have feature_importances_ attribute: {str(e)}")
        return (
            jsonify(
                {
                    "error": "Model does not have feature_importances_ attribute",
                    "details": str(e),
                }
            ),
            500,
        )
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        return jsonify({"error": str(e), "details": str(e)}), 500


@app.route("/feature-names", methods=["GET"])
def get_feature_names():
    """
    Get the feature names used during training.

    Returns:
    JSON response with the feature names.
    """
    logger.info("Getting feature names")
    logger.debug(f"Feature names: {feature_names}")
    return jsonify({"feature_names": feature_names})


@app.route("/model-results", methods=["GET"])
def get_model_results():
    """
    Get the results of each model or a specific model.

    Query parameter:
    - model_name: str, optional, name of the model to query

    Returns:
    JSON response with the results of each model or the specified model.
    """
    results_path = os.path.join(
        os.path.dirname(__file__), "models", "model_results.json"
    )
    logger.debug(f"Results path: {results_path}")
    if not os.path.exists(results_path):
        logger.error(f"Model results file not found at {results_path}")
        return jsonify({"error": "Model results file not found"}), 404

    logger.info(f"Loading model results from {results_path}")
    with open(results_path, "r") as f:
        model_results = json.load(f)

    model_name = request.args.get("model_name")
    if model_name:
        logger.debug(f"Model name: {model_name}")
        if model_name in model_results:
            return jsonify({model_name: model_results[model_name]})
        else:
            return jsonify({"error": f"Model '{model_name}' not found"}), 404

    return jsonify({"model_results": model_results})


@app.route("/training-progress", methods=["GET"])
def get_training_progress():
    """
    Get the training progress.

    Returns:
    JSON response with the training progress.
    """
    progress_path = os.path.join(os.path.dirname(__file__), "models", "progress.json")
    logger.debug(f"Progress path: {progress_path}")
    if not os.path.exists(progress_path):
        return jsonify({"error": "Progress file not found"}), 404

    with open(progress_path, "r") as f:
        progress = json.load(f)

    return jsonify({"progress": progress})


@app.route("/available-models", methods=["GET"])
def get_available_models():
    """
    Get the list of available models from the train_model.yaml configuration.

    Returns:
    JSON response with the list of available models.
    """
    config_path = os.getenv("CONFIG_PATH", os.path.join(BASE_DIR, "train_model.yaml"))
    logger.debug(f"Config path: {config_path}")
    if not os.path.exists(config_path):
        return jsonify({"error": "Configuration file not found"}), 404

    config = load_config(config_path)
    available_models = list(config["models"].keys())
    return jsonify({"available_models": available_models})


@app.route("/class-names", methods=["GET"])
def get_class_names():
    """
    Get the class names from the Failure Type column in the data.

    Returns:
    JSON response with the list of class names.
    """
    data_file_path = os.path.join(
        os.path.dirname(__file__), "data", "predictive_maintenance.csv"
    )
    logger.debug(f"Data file path: {data_file_path}")
    if not os.path.exists(data_file_path):
        logger.error(f"Data file not found at {data_file_path}")
        return jsonify({"error": "Data file not found"}), 404

    df = pd.read_csv(data_file_path)
    if "Failure Type" not in df.columns:
        logger.error("Failure Type column not found in the data")
        return jsonify({"error": "Failure Type column not found in the data"}), 404

    class_names = df["Failure Type"].unique().tolist()
    return jsonify({"class_names": class_names})


@app.route("/retrain", methods=["POST"])
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
            "Logistic Regression": {"C": [0.1, 1, 10]},
            "Decision Tree": {"max_depth": [null, 10, 20, 30]},
            "Random Forest": {"n_estimators": [100, 200], "max_depth": [10, 20]},
            "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]},
            "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
        },
        "models": {
            "Logistic Regression": {"import_module": "sklearn.linear_model", "model_name": "LogisticRegression", "model_params": {"max_iter": 1000}},
            "Decision Tree": {"import_module": "sklearn.tree", "model_name": "DecisionTreeClassifier", "model_params": {"random_state": 42}},
            "Random Forest": {"import_module": "sklearn.ensemble", "model_name": "RandomForestClassifier", "model_params": {"random_state": 42}},
            "Gradient Boosting": {"import_module": "sklearn.ensemble", "model_name": "GradientBoostingClassifier", "model_params": {"random_state": 42}},
            "SVM": {"import_module": "sklearn.svm", "model_name": "SVC", "model_params": {"probability": True}}
        },
        "train_test_split": {"test_size": 0.2, "random_state": 42}
    }

    Returns:
    JSON response with the result of the retraining process.
    """
    config = request.get_json(force=True)
    config_path = os.path.join(BASE_DIR, "retrain_config.json")
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


if __name__ == "__main__":
    logger.info("Starting Flask app")
    app.run(host="0.0.0.0", port=os.environ.get("PORT", 80))
