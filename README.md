# Back-end for Predictive Maintenance System

## Overview

This project is a back-end system for a predictive maintenance application. The system is designed to predict equipment failures based on various sensor data inputs. It includes functionalities for data preprocessing, model training, and serving predictions via a REST API.

## Architectural Overview

The architecture of the predictive maintenance system is designed to be modular and scalable. Below is an overview of the main components:

1. **Data Processing**:
    - **Module**: `src/data_processing/split_data.py`
    - **Functionality**: This module handles the preprocessing of raw data. It splits the dataset into training/validation and testing sets, ensuring that the data is ready for model training and evaluation.
    - **Key Functions**: `split_data`

2. **Model Training**:
    - **Module**: `src/model_training/train_model.py`
    - **Functionality**: This module is responsible for training the predictive model. It loads the preprocessed data, trains the model using machine learning algorithms, and saves the trained model for future use.
    - **Key Functions**: `train_model`

3. **API Serving**:
    - **Module**: `src/api_serving/serve_model.py`
    - **Functionality**: This module serves the trained model via a REST API. It handles incoming prediction requests, processes the input data, and returns the prediction results.
    - **Key Functions**: `predict`, `train_model`

4. **Configuration Management**:
    - **Module**: `src/utils/data_utils.py`
    - **Functionality**: This module manages the configuration settings for the system. It loads configuration parameters from YAML files, making the system flexible and easy to configure.
    - **Key Functions**: `load_config`

5. **Logging and Error Handling**:
    - **Logging**: Comprehensive logging is implemented across all modules using Python's built-in `logging` module. This helps in monitoring the system's operations and debugging issues.
    - **Error Handling**: Robust error handling mechanisms are in place to ensure the system can gracefully handle unexpected situations and provide useful error messages.

6. **Dockerization**:
    - The entire application is containerized using Docker. This ensures consistency across different environments and simplifies deployment. The Docker setup includes a Dockerfile that defines the environment and dependencies required to run the application.
    - Note: While Docker is not used in the primary deployment pipeline, it can be used for independent deployment and testing. This allows developers to run the application in a consistent environment without worrying about local setup issues.

## Major Design Decisions

1. **Modular Design**: 
    - The project is divided into distinct modules for data processing (`src/data_processing/split_data.py`), model training (`src/model_training/train_model.py`), and API serving (`src/api_serving/serve_model.py`). 
    - This separation of concerns ensures that each module can be developed, tested, and maintained independently, improving the overall maintainability and scalability of the system.

2. **Logging**: 
    - Comprehensive logging is implemented using Python's built-in `logging` module. 
    - Different log levels (INFO, DEBUG, ERROR) are used to capture various types of information, from general operational messages to detailed debugging information and error reports. 
    - This facilitates easier debugging and monitoring of the system's operations.

3. **Error Handling**: 
    - Robust error handling mechanisms are in place to ensure the system can gracefully handle unexpected situations. 
    - For example, the `load_config` function in `src/utils/data_utils.py` catches specific exceptions like `FileNotFoundError` and `YAMLError` and logs appropriate error messages. 
    - This prevents the system from crashing and provides useful information for diagnosing issues.

4. **Dockerization**: 
    - The application is containerized using Docker to ensure consistency across different environments and simplify deployment. 
    - Docker allows the application to run in isolated containers, ensuring that it behaves the same way regardless of where it is deployed. 
    - This eliminates issues related to environment differences and makes it easier to manage dependencies.
    - Note: While Docker is not used in the primary deployment pipeline, it can be used for independent deployment and testing. This allows developers to run the application in a consistent environment without worrying about local setup issues.

5. **Configuration Management**: 
    - YAML configuration files are used to manage various settings, making the system flexible and easy to configure. 
    - The `load_config` function in `src/utils/data_utils.py` loads configuration settings from a YAML file, allowing different configurations to be easily applied without changing the code. 
    - This approach enhances the flexibility and adaptability of the system.

## Contributions

1. **Data Processing**: Implemented the `split_data` function to split the dataset into training/validation and testing sets.
2. **Model Training**: Developed the model training script (`src/model_training/train_model.py`) to train and save the predictive model.
3. **API Development**: Created the Flask API (`src/api_serving/serve_model.py`) to serve predictions and handle incoming requests.
4. **Testing**: Wrote unit tests (`tests/test_split_data.py`) to ensure the correctness of the data splitting functionality.
5. **Documentation**: Updated the README with detailed instructions on installation, running tests, and using the API.

## Reason for Choosing This Project

I chose this project because it addresses a real-world problem of predictive maintenance, which is crucial for industries relying on machinery and equipment. The project allowed me to demonstrate my skills in data processing, machine learning, API development, and software engineering best practices. Additionally, it provided an opportunity to work on a comprehensive system that integrates various components, from data handling to model serving.

## Installation

To install the back-end, clone the repository and install the dependencies:
```sh
git clone https://github.com/your-repo/predictive-maintenance-backend.git
cd predictive-maintenance-backend
pip install .
```

## Running Unit Tests

To run the unit tests, ensure the model is trained and the necessary files are in place:
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
```sh
python serve_model.py
```
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
