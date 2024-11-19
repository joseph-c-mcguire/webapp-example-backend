import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from src.ModelMonitor import ModelMonitor

np.random.seed(0)


@pytest.fixture
def sample_data():
    # Create a sample dataset
    X_train = pd.DataFrame(np.random.randn(100, 5), columns=[
                           f'feature_{i}' for i in range(5)])
    y_train = np.random.randint(0, 2, size=100)
    X_new = pd.DataFrame(np.random.randn(20, 5), columns=[
                         f'feature_{i}' for i in range(5)])
    y_true = np.random.randint(0, 2, size=20)
    return X_train, y_train, X_new, y_true


@pytest.fixture
def trained_model(sample_data):
    X_train, y_train, _, _ = sample_data
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def test_model_monitor_initialization(trained_model, sample_data):
    X_train, _, _, _ = sample_data
    monitor = ModelMonitor(trained_model, X_train)
    assert monitor.model == trained_model
    assert monitor.X_train.equals(X_train)
    assert monitor.threshold_drift == 0.05
    assert monitor.train_feature_means.equals(X_train.mean())


def test_calculate_metrics_classification(trained_model, sample_data):
    _, _, X_new, y_true = sample_data
    monitor = ModelMonitor(trained_model, X_new)
    y_pred = trained_model.predict(X_new)
    metrics = monitor.calculate_metrics(
        y_true, y_pred, problem_type='classification')
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics


def test_calculate_metrics_regression():
    # Create a sample regression dataset
    X_train = pd.DataFrame(np.random.randn(100, 5), columns=[
                           f'feature_{i}' for i in range(5)])
    y_train = np.random.randn(100)
    X_new = pd.DataFrame(np.random.randn(20, 5), columns=[
                         f'feature_{i}' for i in range(5)])
    y_true = np.random.randn(20)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    monitor = ModelMonitor(model, X_train)
    y_pred = model.predict(X_new)
    metrics = monitor.calculate_metrics(
        y_true, y_pred, problem_type='regression')
    assert 'mean_squared_error' in metrics


def test_check_feature_drift(trained_model, sample_data):
    X_train, _, X_new, _ = sample_data
    monitor = ModelMonitor(trained_model, X_train)
    drift_detected = monitor.check_feature_drift(X_new)
    assert isinstance(drift_detected, bool)


def test_check_feature_drift_with_drift(trained_model, sample_data):
    X_train, _, _, _ = sample_data
    # Create new data with drift
    X_new = X_train + 1
    monitor = ModelMonitor(trained_model, X_train)
    drift_detected = monitor.check_feature_drift(X_new)
    assert drift_detected


def test_check_feature_drift_without_drift(trained_model, sample_data):
    X_train, _, X_new, _ = sample_data
    monitor = ModelMonitor(trained_model, X_train)
    # Just check that it runs without errors
    drift_detected = monitor.check_feature_drift(X_train)
    assert not drift_detected


def test_monitor_method_classification(trained_model, sample_data):
    _, _, X_new, y_true = sample_data
    monitor = ModelMonitor(trained_model, X_new)
    metrics, drift_detected = monitor.monitor(
        X_new, y_true, problem_type='classification')
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert isinstance(drift_detected, bool)
