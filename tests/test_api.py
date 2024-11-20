
import unittest
import json
from flask import Flask
from serve_model import app


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_predict(self):
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
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('prediction', data)

    def test_predict_probabilities(self):
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
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('probabilities', data)

    def test_get_data(self):
        response = self.app.get('/data')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIsInstance(data, list)

    def test_train(self):
        response = self.app.post('/train')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('message', data)

    def test_get_confusion_matrix(self):
        response = self.app.post('/confusion-matrix')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('confusion_matrix', data)

    def test_get_roc_curve(self):
        response = self.app.post('/roc-curve')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('fpr', data)
        self.assertIn('tpr', data)
        self.assertIn('roc_auc', data)

    def test_get_feature_importance(self):
        response = self.app.get('/feature-importance')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('feature_importance', data)


if __name__ == '__main__':
    unittest.main()
