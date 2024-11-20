import unittest
import json
import logging
from flask import Flask
from serve_model import app

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_predict(self):
        logger.info("Testing /predict endpoint")
        response = self.app.post('/predict', data=json.dumps({
            "features": {
                "Type": "M",
                "Air temperature [K]": 300,
                "Process temperature [K]": 310,
                "Rotational speed [rpm]": 1500,
                "Torque [Nm]": 40,
                "Tool wear [min]": 10
            }
        }), content_type='application/json')
        self.assertEqual(response.status_code, 200, f"Expected status code 200, got {
                         response.status_code}")
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('prediction', data,
                      "Response does not contain 'prediction'")

    def test_predict_probabilities(self):
        logger.info("Testing /predict-probabilities endpoint")
        response = self.app.post('/predict-probabilities', data=json.dumps({
            "data": [
                {
                    "Type": "M",
                    "Air temperature [K]": 300,
                    "Process temperature [K]": 310,
                    "Rotational speed [rpm]": 1500,
                    "Torque [Nm]": 40,
                    "Tool wear [min]": 10
                }
            ]
        }), content_type='application/json')
        self.assertEqual(response.status_code, 200, f"Expected status code 200, got {
                         response.status_code}")
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('probabilities', data,
                      "Response does not contain 'probabilities'")

    def test_get_data(self):
        logger.info("Testing /data endpoint")
        response = self.app.get('/data')
        self.assertEqual(response.status_code, 200, f"Expected status code 200, got {
                         response.status_code}")
        data = json.loads(response.get_data(as_text=True))
        self.assertIsInstance(data, list, "Response data is not a list")

    def test_train(self):
        logger.info("Testing /train endpoint")
        response = self.app.post('/train')
        self.assertIn(response.status_code, [
                      200, 500], f"Expected status code 200 or 500, got {response.status_code}")
        if response.status_code == 500:
            data = json.loads(response.get_data(as_text=True))
            self.assertIn('error', data, "Response does not contain 'error'")
        else:
            data = json.loads(response.get_data(as_text=True))
            self.assertIn('message', data,
                          "Response does not contain 'message'")

    def test_get_confusion_matrix(self):
        logger.info("Testing /confusion-matrix endpoint")
        response = self.app.post('/confusion-matrix')
        self.assertEqual(response.status_code, 200, f"Expected status code 200, got {
                         response.status_code}")
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('confusion_matrix', data,
                      "Response does not contain 'confusion_matrix'")

    def test_get_roc_curve(self):
        logger.info("Testing /roc-curve endpoint")
        response = self.app.post('/roc-curve')
        self.assertEqual(response.status_code, 200, f"Expected status code 200, got {
                         response.status_code}")
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('fpr', data, "Response does not contain 'fpr'")
        self.assertIn('tpr', data, "Response does not contain 'tpr'")
        self.assertIn('roc_auc', data, "Response does not contain 'roc_auc'")

    def test_get_feature_importance(self):
        logger.info("Testing /feature-importance endpoint")
        response = self.app.get('/feature-importance')
        self.assertEqual(response.status_code, 200, f"Expected status code 200, got {
                         response.status_code}")
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('feature_importance', data,
                      "Response does not contain 'feature_importance'")

    def test_get_feature_names(self):
        logger.info("Testing /feature-names endpoint")
        response = self.app.get('/feature-names')
        self.assertEqual(response.status_code, 200, f"Expected status code 200, got {
                         response.status_code}")
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('feature_names', data,
                      "Response does not contain 'feature_names'")


if __name__ == '__main__':
    unittest.main()
