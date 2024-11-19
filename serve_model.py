import logging
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from pandas import DataFrame
import joblib
import pandas as pd

from src.data_utils import load_model, load_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the trained model and ModelMonitor
model_path = os.getenv('MODEL_PATH', r'models/best_model.pkl')
monitor_path = os.getenv('MONITOR_PATH', r'models/model_monitor.pkl')
config_path = os.getenv("CONFIG_PATH", r'config.yaml')
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

    # Monitor model performance
    # metrics, drift_status = monitor.monitor(X_new=features, y_true=prediction)
    # logger.info(f"Prediction metrics: {metrics}")
    # logger.info(f"Drift detected: {drift_status}")

    # JSON doesn't recognize numpy types, so we need to convert the prediction to a Python type
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


# @app.route('/pca-decision-boundary', methods=['POST'])
# def pca_decision_boundary():
#     data = request.get_json(force=True)
#     features = DataFrame(data['data'])

#     # Ensure the input data has the correct number of features
#     for feature in feature_names:
#         if feature not in features.columns:
#             features[feature] = 0

#     # Extract the preprocessor from the pipeline
#     preprocessor = pipeline.named_steps['preprocessor']

#     # Transform the data using the preprocessor
#     transformed_features = preprocessor.transform(features)

#     # Perform PCA on the transformed data
#     pca = PCA(n_components=2)
#     pca_result = pca.fit_transform(transformed_features)

#     # Generate a mesh grid based on the min and max values from the training data
#     x_min, x_max = float(min_max_values['min'][feature_names[0]]), float(
#         min_max_values['max'][feature_names[0]])
#     y_min, y_max = float(min_max_values['min'][feature_names[1]]), float(
#         min_max_values['max'][feature_names[1]])
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
#                          np.arange(y_min, y_max, 0.1))

#     # Evaluate the model's decision function on the grid
#     grid_points = np.c_[xx.ravel(), yy.ravel()]
#     grid_probabilities = pipeline.predict_proba(
#         pca.inverse_transform(grid_points))[:, 1]
#     zz = grid_probabilities.reshape(xx.shape)

#     return jsonify({
#         'pca_data': pca_result.tolist(),
#         'decision_boundary': {
#             'x': xx.tolist(),
#             'y': yy.tolist(),
#             'z': zz.tolist()
#         }
#     })


if __name__ == '__main__':
    logger.info("Starting Flask app")
    app.run(host='0.0.0.0', port=5000)
