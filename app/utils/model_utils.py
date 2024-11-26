import logging
import importlib
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
import joblib
from pandas import to_pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def select_features(
    X: pd.DataFrame, y: pd.Series, preprocessor: ColumnTransformer
) -> List[int]:
    """
    Select the most relevant features using a Random Forest model.

    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Target variable.
    preprocessor : ColumnTransformer
        Preprocessor for the data.

    Returns
    -------
    List[int]
        Selected feature indices.

    Examples
    --------
    >>> selected_features = select_features(X, y, preprocessor)
    >>> print(selected_features)
    """
    logger.info("Starting feature selection")
    processed_data = preprocessor.fit_transform(X)
    model = RandomForestClassifier()
    model.fit(processed_data, y)

    # Get feature importances
    selected_features = get_top_n_indices(model.feature_importances_)
    logger.info("Feature selection completed")
    return selected_features


def train_and_evaluate_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: Dict[str, BaseEstimator],
) -> Dict[str, Dict[str, Any]]:
    """
    Train and evaluate multiple machine learning models.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training target.
    X_test : np.ndarray
        Testing features.
    y_test : np.ndarray
        Testing target.
    models : Dict[str, BaseEstimator]
        Dictionary of model names and their corresponding estimators.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Model evaluation results.

    Examples
    --------
    >>> models = {
    >>>     'Logistic Regression': LogisticRegression(),
    >>>     'Decision Tree': DecisionTreeClassifier(),
    >>>     'Random Forest': RandomForestClassifier(),
    >>>     'Gradient Boosting': GradientBoostingClassifier(),
    >>>     'SVM': SVC(probability=True)
    >>> }
    >>> results = train_and_evaluate_model(X_train, y_train, X_test, y_test, models)
    >>> for model_name, result in results.items():
    >>>     print(f"Model: {model_name}")
    >>>     print(result['Classification Report'])
    >>>     print(f"ROC AUC: {result['ROC AUC']}\n")
    """
    logger.info("Starting model training and evaluation")
    results = {}

    for name, model in models.items():
        logger.info(f"Training {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "Classification Report": classification_report(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_proba),
        }
        logger.info(f"Evaluation of {name} completed")

    logger.info("Model training and evaluation completed")
    return results


def tune_model(
    models: Dict[str, BaseEstimator],
    param_grids: Dict[str, dict],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> BaseEstimator:
    """
    Perform hyperparameter tuning using GridSearchCV for multiple models.

    Parameters
    ----------
    models : Dict[str, BaseEstimator]
        Dictionary of model names and their corresponding estimators.
    param_grids : Dict[str, dict]
        Dictionary of model names and their corresponding hyperparameter grids.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.

    Returns
    -------
    BaseEstimator
        Best estimator after hyperparameter tuning.

    Examples
    --------
    >>> models = {
    >>>     'Random Forest': RandomForestClassifier(),
    >>>     'Gradient Boosting': GradientBoostingClassifier()
    >>> }
    >>> param_grids = {
    >>>     'Random Forest': {'n_estimators': [100, 200], 'max_depth': [10, 20]},
    >>>     'Gradient Boosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
    >>> }
    >>> best_model = tune_model(models, param_grids, X_train, y_train)
    """
    logger.info("Starting hyperparameter tuning")
    best_model = None
    best_score = -np.inf

    for name, model in models.items():
        logger.info(f"Tuning {name}")
        param_grid = param_grids.get(name, {})
        grid_search = GridSearchCV(
            estimator=model, param_grid=param_grid, cv=5, scoring="roc_auc", n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_

        logger.info(f"Best score for {name}: {grid_search.best_score_}")

    logger.info("Hyperparameter tuning completed")
    return best_model


def interpret_model(model: BaseEstimator, X: pd.DataFrame) -> None:
    """
    Interpret the machine learning model using SHAP values.

    Parameters
    ----------
    model : BaseEstimator
        Trained machine learning model.
    X : pd.DataFrame
        Features for interpretation.

    Returns
    -------
    None

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> model = RandomForestClassifier().fit(X_train, y_train)
    >>> interpret_model(model, X_test)
    """
    logger.info(f"Starting model interpretation for {model.__class__.__name__}")
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # Summary plot
    shap.summary_plot(shap_values, X)
    logger.info(f"Model interpretation for {model.__class__.__name__} completed")


def monitor_model_performance(
    model: BaseEstimator, X: pd.DataFrame, y: pd.Series
) -> float:
    """
    Monitor the performance of the deployed model.

    Parameters
    ----------
    model : BaseEstimator
        Trained machine learning model.
    X : pd.DataFrame
        Features.
    y : pd.Series
        Target variable.

    Returns
    -------
    float
        ROC AUC score of the model on new data.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> model = RandomForestClassifier().fit(X_train, y_train)
    >>> roc_auc = monitor_model_performance(model, X_test, y_test)
    >>> print(roc_auc)
    """
    logger.info(f"Monitoring performance of {model.__class__.__name__}")
    y_proba = model.predict_proba(X)[:, 1]
    roc_auc = roc_auc_score(y, y_proba)
    logger.info(f"Performance monitoring completed with ROC AUC: {roc_auc}")

    return roc_auc


def get_top_n_indices(arr: np.ndarray, n: int = 10) -> np.ndarray:
    """
    Get the indices of the n largest elements from a NumPy array.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    n : int, optional
        Number of top elements to retrieve indices for, by default 10.

    Returns
    -------
    np.ndarray
        Array of indices of the n largest elements.

    Examples
    --------
    >>> arr = np.array([1, 3, 5, 7, 9, 2, 4, 6, 8, 0])
    >>> top_indices = get_top_n_indices(arr, 3)
    >>> print(top_indices)
    """
    logger.info(f"Getting top {n} indices from array")
    if n >= len(arr):
        return np.argsort(arr)[-n:][::-1]

    # Get the indices that would sort the array
    sorted_indices = np.argsort(arr)
    # Select the last n indices and reverse them
    top_n_indices = sorted_indices[-n:][::-1]
    logger.info(f"Top {n} indices obtained")

    return top_n_indices


def save_model(model: BaseEstimator, file_path: str) -> None:
    """
    Save the trained model to a file.

    Parameters
    ----------
    model : BaseEstimator
        Trained machine learning model.
    file_path : str
        Path to save the model.

    Returns
    -------
    None

    Examples
    --------
    >>> save_model(model, 'best_model.pkl')
    """
    logger.info(f"Saving model to {file_path}")
    to_pickle(model, file_path)
    logger.info("Model saved successfully")


def load_model(file_path: str) -> BaseEstimator:
    """
    Load a trained model from a file.

    Parameters
    ----------
    file_path : str
        Path to the model file.

    Returns
    -------
    BaseEstimator
        Loaded model.

    Examples
    --------
    >>> model = load_model('best_model.pkl')
    """
    logger.info(f"Loading model from {file_path}")
    model = joblib.load(file_path)
    logger.info("Model loaded successfully")
    return model


def get_model(model_name: str, model_params: Dict[str, Any]) -> BaseEstimator:
    """
    Get a machine learning model instance based on the model name and parameters.

    Parameters
    ----------
    model_name : str
        Full path of the model class (e.g., 'sklearn.ensemble.RandomForestClassifier').
    model_params : Dict[str, Any]
        Parameters to initialize the model.

    Returns
    -------
    BaseEstimator
        Initialized machine learning model.

    Raises
    ------
    Exception
        If there is an error loading the model.

    Examples
    --------
    >>> model = get_model('sklearn.ensemble.RandomForestClassifier', {'n_estimators': 100})
    """
    try:
        module_name, class_name = model_name.rsplit(".", 1)
        model_class = getattr(importlib.import_module(module_name), class_name)
        return model_class(**model_params)
    except (AttributeError, ValueError, ImportError) as e:
        logger.error(f"Error loading model {model_name}: {e}")
        raise
