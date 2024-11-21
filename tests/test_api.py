import os
import sys
import unittest
import json
from flask import Flask
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from serve_model import app


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def generate_random_features(self):
        return {
            "Type": random.choice(["M", "L", "H"]),
            "Air temperature [K]": random.uniform(290, 310),
            "Process temperature [K]": random.uniform(300, 320),
            "Rotational speed [rpm]": random.uniform(1000, 2000),
            "Torque [Nm]": random.uniform(20, 50),
            "Tool wear [min]": random.uniform(0, 300),
        }

    def test_predict(self):
        response = self.app.post(
            "/predict",
            data=json.dumps({"features": self.generate_random_features()}),
            content_type="application/json",
        )
        self.assertEqual(
            response.status_code,
            200,
            msg=f"Response data: {response.get_data(as_text=True)}",
        )
        data = json.loads(response.get_data(as_text=True))
        self.assertIn("prediction", data)

    def test_predict_probabilities(self):
        response = self.app.post(
            "/predict-probabilities",
            data=json.dumps({"data": [self.generate_random_features()]}),
            content_type="application/json",
        )
        self.assertEqual(
            response.status_code,
            200,
            msg=f"Response data: {response.get_data(as_text=True)}",
        )
        data = json.loads(response.get_data(as_text=True))
        self.assertIn("probabilities", data)

    def test_get_data(self):
        response = self.app.get("/data")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIsInstance(data, list)

    def test_get_confusion_matrix(self):
        # Define the required JSON payload
        payload = {"model_name": "Decision Tree", "class_label": "No Failure"}

        # Send the POST request with the JSON payload
        response = self.app.post(
            "/confusion-matrix",
            data=json.dumps(payload),
            content_type="application/json",
        )

        self.assertEqual(
            response.status_code,
            200,
            msg=f"Response data: {response.get_data(as_text=True)}",
        )

        data = json.loads(response.get_data(as_text=True))
        self.assertIn("confusion_matrix", data)

    def test_get_roc_curve(self):
        # Define the required JSON payload for ROC curve
        payload = {"model_name": "Decision Tree", "class_label": "No Failure"}

        # Send the POST request with the JSON payload
        response = self.app.post(
            "/roc-curve", data=json.dumps(payload), content_type="application/json"
        )

        self.assertEqual(
            response.status_code,
            200,
            msg=f"Response data: {response.get_data(as_text=True)}",
        )

        data = json.loads(response.get_data(as_text=True))
        self.assertIn("fpr", data)
        self.assertIn("tpr", data)
        self.assertIn("roc_auc", data)

    def test_get_feature_importance(self):
        response = self.app.get("/feature-importance")
        self.assertEqual(
            response.status_code,
            200,
            msg=f"Response data: {response.get_data(as_text=True)}",
        )
        data = json.loads(response.get_data(as_text=True))
        self.assertIn("feature_importance", data)


if __name__ == "__main__":
    unittest.main()
