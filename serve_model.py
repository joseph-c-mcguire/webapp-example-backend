import logging
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from pandas import DataFrame
import joblib
import pandas as pd
import subprocess
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split

from src.data_utils import load_model, load_config
from split_data import split_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the trained model and ModelMonitor
model_path = os.getenv('MODEL_PATH', r'models/best_model.pkl')
monitor_path = os.getenv('MONITOR_PATH', r'models/model_monitor.pkl')
config_path = os.getenv("CONFIG_PATH", r'train_model.yaml')
logger.info(f"Loading the trained model from {model_path}")
pipeline = load_model(model_path)
logger.info(f"Loading the ModelMonitor from {monitor_path}")
monitor = joblib.load(monitor_path)
logger.info(f"Loading the Configuration File from {config_path}")
config = load_config(config_path)

# Load the min and max values from the training data
min_max_values_path = os.getenv(
    'MIN_MAX_VALUES_PATH', r'models/min_max_values.pkl')
logger.info(f"Loading the min and max values from {min_max_values_path}")
min_max_values = joblib.load(min_max_values_path)

# Get the feature names used during training
feature_names = pipeline.named_steps['preprocessor'].transformers_[0][2] + \
    pipeline.named_steps['preprocessor'].transformers_[1][2] + \
    pipeline.named_steps['preprocessor'].transformers_[2][2]


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict equipment failure using the deployed model.

    Request JSON format:
    {
        "features": {
            "Type": "value",
            "Air temperature [K]": value,
            "Process temperature [K]": value,
            "Rotational speed [rpm]": value,
            "Torque [Nm]": value,
            "Tool wear [min]": value
        }
    }

    Returns:
    JSON response with prediction result.
    """
    data = request.get_json(force=True)
    features = DataFrame(data['features'], index=[0])
    prediction = pipeline.predict(features)

    # Map prediction to "Failed" or "Not Failed"
    prediction_label = "Failed" if prediction[0] == 1 else "Not Failed"

    return jsonify({'prediction': prediction_label})


@app.route('/predict-probabilities', methods=['POST'])
def predict_probabilities():
    data = request.get_json(force=True)
    features = DataFrame(data['data'])
    probabilities = pipeline.predict_proba(
        features)[:, 1]  # Assuming binary classification
    return jsonify({'probabilities': probabilities.tolist()})


@app.route('/data', methods=['GET'])
def get_data():
    data_file_path = os.path.join(os.path.dirname(
        __file__), 'data', 'predictive_maintenance.csv')
    if not os.path.exists(data_file_path):
        logger.error(f"Data file not found at {data_file_path}")
        return jsonify({'error': 'Data file not found'}), 404
    df = pd.read_csv(data_file_path)
    data = df.to_dict(orient='records')
    return jsonify(data)


@app.route('/train', methods=['POST'])
def train_model():
    """
    Train the model by calling the train_model.py script.

    Returns:
    JSON response with the result of the training process.
    """
    train_val_file_path = os.path.join(
        os.path.dirname(__file__), 'data', 'train_val_data.csv')
    test_file_path = os.path.join(
        os.path.dirname(__file__), 'data', 'test_data.csv')

    # Check if the train and test data files exist
    if not os.path.exists(train_val_file_path) or not os.path.exists(test_file_path):
        logger.info(
            "Train or test data file not found. Running split_data function to generate them.")
        try:
            split_data(
                data_file_path=os.path.join(os.path.dirname(
                    __file__), 'data', 'predictive_maintenance.csv'),
                train_val_file_path=train_val_file_path,
                test_file_path=test_file_path
            )
        except Exception as e:
            logger.error(
                f"An error occurred while splitting the data: {str(e)}")
            return jsonify({'error': 'An error occurred while splitting the data', 'details': str(e)}), 500

    # Proceed with training the model
    try:
        logger.info("Starting model training")
        result = subprocess.run(
            ['python', 'train_model.py'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("Model training completed successfully")
            return jsonify({'message': 'Model training completed successfully'}), 200
        else:
            logger.error(f"Model training failed: {result.stderr}")
            return jsonify({'error': 'Model training failed', 'details': result.stderr}), 500
    except Exception as e:
        logger.error(f"An error occurred while training the model: {str(e)}")
        return jsonify({'error': 'An error occurred while training the model', 'details': str(e)}), 500


@app.route('/confusion-matrix', methods=['POST'])
def get_confusion_matrix():
    """
    Get the confusion matrix for the model using test data from test_data.csv.

    Returns:
    JSON response with the confusion matrix.
    """
    test_file_path = os.path.join(
        os.path.dirname(__file__), 'data', 'test_data.csv')
    if not os.path.exists(test_file_path):
        logger.error(f"Test data file not found at {test_file_path}")
        return jsonify({'error': 'Test data file not found'}), 404

    logger.info(f"Loading test data from {test_file_path}")
    df = pd.read_csv(test_file_path)
    # Assuming 'Target' is the label column
    features = df.drop('Target', axis=1)
    labels = df['Target']

    logger.info("Calculating confusion matrix")
    predictions = pipeline.predict(features)
    cm = confusion_matrix(labels, predictions)
    return jsonify({'confusion_matrix': cm.tolist()})


@app.route('/roc-curve', methods=['POST'])
def get_roc_curve():
    """
    Generate the ROC curve data for the model using test data from test_data.csv.

    Returns:
    JSON response with the false positive rate, true positive rate, and AUC.
    """
    test_file_path = os.path.join(
        os.path.dirname(__file__), 'data', 'test_data.csv')
    if not os.path.exists(test_file_path):
        logger.error(f"Test data file not found at {test_file_path}")
        return jsonify({'error': 'Test data file not found'}), 404

    logger.info(f"Loading test data from {test_file_path}")
    df = pd.read_csv(test_file_path)
    # Assuming 'Target' is the label column
    features = df.drop('Target', axis=1)
    labels = df['Target']

    logger.info("Generating ROC curve data")
    probabilities = pipeline.predict_proba(features)[:, 1]
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)

    return jsonify({'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'roc_auc': roc_auc})


@app.route('/feature-importance', methods=['GET'])
def get_feature_importance():
    """
    Get the feature importance of the model.

    Returns:
    JSON response with the feature importance.
    """
    logger.info("Getting feature importance")
    try:
        # Assuming the model has a feature_importances_ attribute
        feature_importances = pipeline.named_steps['model'].feature_importances_
        feature_importance_dict = dict(zip(feature_names, feature_importances))
        return jsonify({'feature_importance': feature_importance_dict})
    except AttributeError as e:
        logger.error(
            f"Model does not have feature_importances_ attribute: {str(e)}")
        return jsonify({'error': 'Model does not have feature_importances_ attribute', 'details': str(e)}), 500


if __name__ == '__main__':
    logger.info("Starting Flask app")
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 80))
