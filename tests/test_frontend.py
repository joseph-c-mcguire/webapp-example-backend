import unittest
import requests
import json
import os


class TestFrontend(unittest.TestCase):

    def setUp(self):
        self.frontend_url = os.getenv(
            "FRONTEND_URL",
            "https://webapp-example-frontend-56f2ec31cf0a.herokuapp.com/",
        )

    def test_frontend_availability(self):
        response = requests.get(self.frontend_url)
        self.assertEqual(
            response.status_code,
            200,
            msg=f"Frontend is not available. Status code: {response.status_code}",
        )


if __name__ == "__main__":
    unittest.main()
