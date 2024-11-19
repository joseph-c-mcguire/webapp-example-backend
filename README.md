# Back-end for Predictive Maintenance System

## Installation

To install the back-end, clone the repository and install the dependencies.

## Running Unit Tests

To run the unit tests, use the following command:
```sh
pytest
```

### Troubleshooting Unit Tests

If you encounter issues while running the unit tests, consider the following steps:

1. Ensure all dependencies are installed correctly.
2. Check for any syntax errors or typos in your test files.
3. Verify that the paths to any required files or resources are correct.
4. Review the error messages for clues on what might be going wrong.
5. If the issue persists, try running the tests with increased verbosity:
```sh
pytest -v
```

## Running the API
To run the API, follow these steps:

1. Ensure you have the trained model and ModelMonitor saved in the specified paths. By default, these are models/best_model.pkl and models/model_monitor.pkl.

2. Set the environment variables for the model and monitor paths if they are different from the default:
```sh
export MODEL_PATH=path/to/your/best_model.pkl
export MONITOR_PATH=path/to/your/model_monitor.pkl
export CONFIG_PATH=path/to/your/config.yaml
```

3. Start the Flask API:
The API will be available at http://0.0.0.0:5000.

## Running the API via Docker
To run the API using Docker, follow these steps:

1. Build the Docker image:
```sh
docker build -t predictive-maintenance-api .
```

2. Run the Docker container:
```sh
docker run -p 5000:5000 -e MODEL_PATH=path/to/your/best_model.pkl -e MONITOR_PATH=path/to/your/model_monitor.pkl -e CONFIG_PATH=path/to/your/config.yaml predictive-maintenance-api
```

The API will be available at http://0.0.0.0:5000.

## API Endpoints
Predict Equipment Failure
**Endpoint**: /predict

**Method**: POST

### Request JSON format:
```json
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
```

### Response JSON format:
```json
{
    "prediction": "Failed" or "Not Failed"
}
```

### Example Request:
```curl
curl -X POST http://0.0.0.0:5000/predict -H "Content-Type: application/json" -d '{
    "features": {
        "Type": "M",
        "Air temperature [K]": 300,
        "Process temperature [K]": 310,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40,
        "Tool wear [min]": 10
    }
}'
```
### Example Response:
```json
{
    "prediction": "Not Failed"
}
```

## Running the Model Training Script

To train the model, use the following command:
```sh
python train_model.py
```

## Running the Model Serving Script

To serve the model, use the following command:
```sh
python serve_model.py
```