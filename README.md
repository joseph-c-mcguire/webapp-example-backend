# Back-end for Predictive Maintenance System

## Overview

This project is a backend system for a predictive maintenance application.
The system is designed for training, serving, and producing diagnostics for a set of models trained on a UCI dataset, more info can be found here: [https://webapp-example-frontend-56f2ec31cf0a.herokuapp.com/](https://webapp-example-frontend-56f2ec31cf0a.herokuapp.com/)
This was a learning exercise to build a full-stack web application, the frontend repo can be found here: [https://github.com/joseph-c-mcguire/webapp-example-frontend](https://github.com/joseph-c-mcguire/webapp-example-frontend).

## Description
The data-set in question is a [predictive maintenance](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset) hosted by the University of California, Irvine, and can also be found on Kaggle.com.
I trained 3 sample models, which can be configured with the train_model.yaml file, to deal with the multi-class classification problem.
Namely Decision Tree, Gradient Boosting, and Random Forest models.
These were selected due to them all sharing the same class structure in Scikit-Learn, and being good for comparison as they're all decision-tree based models or ensembles of decision-trees.
So this made the retrieving of attributes like `feature_importances_`, and methods like `predict_proba` consistent across the different models.

The backend is built on a Flask server, you can run it locally by running `run.py` and the production server uses `app.py`.
The main difference is that `run.py` has debug enabled, while `app.py` does not.
Also `app.py` deals with cross-browser functionality by using the CORS extension of Flask.

## Design Highlights

- **Extendable:** With the structure of the routes and services for the API, one can easily add new functionalities on to the API, and add end-points to access them.
- **Micro-Services Ready:** As this is set now, separate services can be containerized and orchestrator through Docker and Docker-Compose to build a scalable micro-services platform.
- **Documentation:** Through documentation is included in doc-strings and code comments to ensure readability and accessibility for new developers, if they're interested in using this as a template to build on.
- **Testing:** Unit testing is provided to cover a majority of the codebase, and to be expanded upon in the future, to ensure functionality between the front-end and this back-end API is in continuous sync. A CI pipeline ensures that pushes and merges are maintaining functionality at development time.

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

## Repository Structure

### `run.py` & `app.py`
`run.py` is the development server call function, while `app.py` is the production server.

### `app/`

Holds all the source code for the project

#### `app/models/`
Holds the `model_manager` class, which deals with the loading and saving of the models.

#### `app/routes/`
Holds the Flask routing for different services, specifying the endpoints, routes and any request handling and logging here.

#### `app/services/`
Holds the main logic of the API, and the internal logic behind the endpoints, including helper, serving, training and diagnostic functions.

#### `app/utils/`
Holds miscellaneous utility functions

#### `app/config.py`
Deals with environmental variables, like file-paths, training parameters, data set parameters, etc.

#### `app/extensions.py`
Deals with adding Flask extensions like CORS.

### `data/`
Holds the processed and raw data files for the project

### `models/`
Holds the trained models for the project that are used by the frontend for serving.

### `scripts/`
Currently just holds one script that has a sample CURL request for using the API.

### `tests/`
Holds the unit tests for the repository.

## API Documentation

### Confusion Matrix
**Endpoint**: `/api/diagnostics/confusion-matrix`

**Method**: `POST`

**Description**: Retrieves the confusion matrix for the specified model and class label using test data.

**Request JSON format**:
```json
{
    "model_name": "string",
    "class_label": "string"
}
```

**Response JSON format**:
```json
{
    "confusion_matrix": [
        [true_negative, false_positive],
        [false_negative, true_positive]
    ]
}
```

### ROC Curve
**Endpoint**: `/api/diagnostics/roc-curve`

**Method**: `POST`

**Description**: Retrieves the class-specific ROC curve data for the specified model using test data.

**Request JSON format**:
```json
{
    "model_name": "string",
    "class_name": "string"
}
```

**Response JSON format**:
```json
{
    "fpr": [list_of_false_positive_rates],
    "tpr": [list_of_true_positive_rates],
    "thresholds": [list_of_thresholds]
}
```

### Feature Importance
**Endpoint**: `/api/diagnostics/feature-importance`

**Method**: `POST`

**Description**: Retrieves the feature importance scores for the specified model.

**Request JSON format**:
```json
{
    "model_name": "string"
}
```

**Response JSON format**:
```json
{
    "feature_importance": {
        "feature_name_1": importance_score_1,
        "feature_name_2": importance_score_2,
        ...
    }
}
```

### Data Endpoint
**Endpoint**: `api/helper/data`

**Method**: `GET`

**Description**: Retrieves data from the predictive maintenance CSV file.

**Response**:
- Returns a JSON array of data records.

### Feature Names
**Endpoint**: `api/helper/feature-names`

**Method**: `GET`

**Description**: Retrieves the feature names used during training.

**Response JSON format**:
```json
{
    "feature_names": ["feature1", "feature2", ...]
}
```

### Available Models
**Endpoint**: `/api/helper/available-models`

**Method**: `GET`

**Description**: Retrieves the list of available models from the configuration.

**Response JSON format**:
```json
{
    "available_models": ["model1", "model2", ...]
}
```

### Training Progress
**Endpoint**: `/api/helper/training-progress`

**Method**: `GET`

**Description**: Retrieves the training progress of the current model.


### Model Results
**Endpoint**: `/api/helper/model-results`

**Method**: `GET`

**Description**: Retrieves the results of the trained model.

**Response JSON format**:
```json
{
    "model_results": {
        "accuracy": model_accuracy,
        "precision": model_precision,
        "recall": model_recall,
        "f1_score": model_f1_score
    }
}
```

### Predict Probabilities
**Endpoint**: `/predict-probabilities`

**Method**: `POST`

**Description**: Predicts the probabilities of equipment failure types using the specified model.

**Request JSON format**:
```json
{
    "model_name": "string",
    "data": {
        "sensor1": value1,
        "sensor2": value2,
        ...
    }
}
```

**Response JSON format**:
```json
{
    "probabilities": {
        "failure_type_1": probability_1,
        "failure_type_2": probability_2,
        ...
    }
}
```
