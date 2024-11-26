from app.utils.data_utils import (
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

    def test_get_top_n_indices(self):
        arr = np.array([1, 3, 5, 7, 9, 2, 4, 6, 8, 0])
        top_indices = get_top_n_indices(arr, 3)
        self.assertEqual(list(top_indices), [4, 8, 3])

    def test_get_model(self):
        model = get_model(
            "sklearn.ensemble.RandomForestClassifier", {"n_estimators": 10}
        )
        self.assertIsInstance(model, RandomForestClassifier)

        class PredictiveMaintenanceTests(unittest.TestCase):
            # ...

            def test_get_top_n_indices_basic(self):
                arr = np.array([1, 3, 5, 7, 9, 2, 4, 6, 8, 0])
                top_indices = get_top_n_indices(arr, 3)
                self.assertEqual(list(top_indices), [4, 8, 3])

            def test_get_top_n_indices_all_elements(self):
                arr = np.array([1, 3, 5, 7, 9, 2, 4, 6, 8, 0])
                top_indices = get_top_n_indices(arr, 10)
                self.assertEqual(list(top_indices), [4, 8, 3, 7, 6, 2, 1, 5, 0, 9])

            def test_get_top_n_indices_more_than_length(self):
                arr = np.array([1, 3, 5, 7, 9, 2, 4, 6, 8, 0])
                top_indices = get_top_n_indices(arr, 15)
                self.assertEqual(list(top_indices), [4, 8, 3, 7, 6, 2, 1, 5, 0, 9])

            def test_get_top_n_indices_with_duplicates(self):
                arr = np.array([1, 3, 5, 7, 9, 2, 4, 6, 8, 9])
                top_indices = get_top_n_indices(arr, 3)
                self.assertEqual(list(top_indices), [9, 4, 8])

            def test_get_top_n_indices_single_element(self):
                arr = np.array([1])
                top_indices = get_top_n_indices(arr, 1)
                self.assertEqual(list(top_indices), [0])

            def test_get_top_n_indices_empty_array(self):
                arr = np.array([])
                top_indices = get_top_n_indices(arr, 3)
                self.assertEqual(list(top_indices), [])

            def test_get_top_n_indices_n_is_zero(self):
                arr = np.array([1, 3, 5, 7, 9, 2, 4, 6, 8, 0])
                top_indices = get_top_n_indices(arr, 0)
                self.assertEqual(list(top_indices), [])


if __name__ == "__main__":
    unittest.main()
