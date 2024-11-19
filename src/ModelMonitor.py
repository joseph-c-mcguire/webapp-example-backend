import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelMonitor:
    def __init__(self, model: BaseEstimator, X_train: pd.DataFrame, threshold_drift=0.05):
        self.model = model
        self.X_train = X_train
        self.threshold_drift = threshold_drift
        # Store initial training mean values for feature drift detection
        self.train_feature_means = X_train.mean()

    def calculate_metrics(self, y_true, y_pred, problem_type='classification'):
        if problem_type == 'classification':
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            metrics = {'accuracy': accuracy, 'precision': precision,
                       'recall': recall, 'f1_score': f1}
        elif problem_type == 'regression':
            mse = mean_squared_error(y_true, y_pred)
            metrics = {'mean_squared_error': mse}
        else:
            raise ValueError(
                "Unknown problem type. Use 'classification' or 'regression'.")

        return metrics

    def check_feature_drift(self, X_new):
        # Compare feature means to detect drift
        new_means = X_new.mean()
        drift = (new_means - self.train_feature_means).abs() / \
            self.train_feature_means.abs()
        drift_detected = drift > self.threshold_drift
        drift_report = drift[drift_detected]

        if not drift_report.empty:
            logger.warning("Feature drift detected in features: %s",
                           drift_report.index.tolist())
        return bool(drift_detected.any())

    def monitor(self, X_new, y_true, problem_type='classification'):
        # Generate predictions
        y_pred = self.model.predict(X_new)

        # Calculate performance metrics
        metrics = self.calculate_metrics(y_true, y_pred, problem_type)
        logger.info("Model performance metrics: %s", metrics)

        # Check for feature drift
        drift_detected = self.check_feature_drift(X_new)

        # Log a summary of the monitoring results
        if drift_detected:
            logger.warning(
                "Action required: Significant feature drift detected.")

        # Return the performance metrics and drift status
        return metrics, drift_detected
