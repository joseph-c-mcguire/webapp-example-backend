from src.data_utils import (
    load_config,
    load_data,
    preprocess_data,
    perform_eda,
    select_features,
    train_and_evaluate_model,
    tune_model,
    interpret_model,
    monitor_model_performance,
    get_top_n_indices,
    get_model,
)
import sys
import unittest
from unittest.mock import patch, MagicMock
import os

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import importlib

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

np.random.seed(42)


class PredictiveMaintenanceTests(unittest.TestCase):
    @patch("builtins.open")
    @patch("yaml.load")
    def test_load_config(self, mock_load, mock_open):
        mock_load.return_value = {"param": "value"}
        config = load_config("config.yaml")
        self.assertEqual(config, {"param": "value"})

    @patch("pandas.read_csv")
    def test_load_data(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        data = load_data("data.csv")
        self.assertTrue(isinstance(data, pd.DataFrame))
        self.assertEqual(data.shape, (2, 2))

    def test_preprocess_data(self):
        data = pd.DataFrame(
            {"drop_col": [1, 2], "scale_col": [3, 4], "encode_col": ["A", "B"]}
        )
        preprocessor, processed_data = preprocess_data(
            data,
            columns_to_drop=["drop_col"],
            columns_to_scale=["scale_col"],
            columns_to_encode=["encode_col"],
        )
        self.assertIsNotNone(preprocessor)
        self.assertEqual(processed_data.shape[1], 3)  # scale + encode columns

    @patch("matplotlib.pyplot.show")
    def test_perform_eda(self, mock_show):
        data = pd.DataFrame(
            {"col1": np.random.randn(100), "col2": np.random.randn(100)}
        )
        # No assertion needed, but it should run without errors
        perform_eda(data)
        self.assertTrue(mock_show.called)

    def test_select_features(self):
        X = pd.DataFrame(np.random.randn(100, 4), columns=["a", "b", "c", "d"])
        y = pd.Series(np.random.randint(0, 2, size=100))
        preprocessor = MagicMock()
        preprocessor.fit_transform.return_value = X.values
        selected_features = select_features(X, y, preprocessor)
        self.assertEqual(len(selected_features), 4)

    def test_train_and_evaluate_model(self):
        X_train = np.random.randn(50, 4)
        y_train = np.random.randint(0, 2, size=50)
        X_test = np.random.randn(20, 4)
        y_test = np.random.randint(0, 2, size=20)
        results = train_and_evaluate_model(
            X_train,
            y_train,
            X_test,
            y_test,
            {"Logistic Regression": LogisticRegression()},
        )
        self.assertTrue(isinstance(results, dict))
        self.assertIn("Logistic Regression", results)
        self.assertIn("Classification Report", results["Logistic Regression"])

    def test_tune_model(self):
        X_train = np.random.randn(50, 4)
        y_train = np.random.randint(0, 2, size=50)
        model = {"Random Forest": RandomForestClassifier()}
        param_grid = {"n_estimators": [10, 50], "max_depth": [2, 5]}
        best_model = tune_model(model, param_grid, X_train, y_train)
        self.assertIsInstance(best_model, RandomForestClassifier)

    @patch("shap.Explainer")
    @patch("shap.summary_plot")
    def test_interpret_model(self, mock_summary_plot, mock_explainer):
        X = pd.DataFrame(np.random.randn(20, 4), columns=["a", "b", "c", "d"])
        model = RandomForestClassifier()
        model.fit(X, np.random.randint(0, 2, size=20))
        interpret_model(model, X)
        self.assertTrue(mock_summary_plot.called)
        self.assertTrue(mock_explainer.called)

    def test_monitor_model_performance(self):
        X = np.random.randn(50, 4)
        y = np.random.randint(0, 2, size=50)
        model = RandomForestClassifier()
        model.fit(X, y)
        auc_score = monitor_model_performance(model, X, y)
        self.assertTrue(0 <= auc_score <= 1)

    def test_get_top_n_indices(self):
        arr = np.array([1, 3, 5, 7, 9, 2, 4, 6, 8, 0])
        top_indices = get_top_n_indices(arr, 3)
        self.assertEqual(list(top_indices), [4, 8, 3])

    def test_get_model(self):
        model = get_model(
            "sklearn.ensemble.RandomForestClassifier", {"n_estimators": 10}
        )
        self.assertIsInstance(model, RandomForestClassifier)


if __name__ == "__main__":
    unittest.main()
