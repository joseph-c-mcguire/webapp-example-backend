"""
This module initializes and runs the Flask application.

It creates the Flask app using the application factory pattern, applies Cross-Origin Resource Sharing (CORS)
settings, and runs the app when executed as the main program.
"""

from app import create_app  # Import the application factory function
from flask_cors import CORS  # Import the Flask-CORS extension
from flask import request, jsonify  # Import the Flask request and jsonify functions
import os  # Import the os module to access environment variables

# Create the Flask application instance
app = create_app()

# Enable CORS for the app, allowing requests from 'http://localhost' to '/api/*' endpoints
CORS(app, resources={r"/api/*": {"origins": "http://localhost"}})

# Define the secret token from environment variables
SECRET_TOKEN = os.getenv("SECRET_TOKEN")


def verify_token(token):
    return token == SECRET_TOKEN


@app.before_request
def authenticate():
    token = request.headers.get("Authorization")
    if not token or not verify_token(token):
        return jsonify({"message": "Unauthorized"}), 401


if __name__ == "__main__":
    # Run the Flask development server
    app.run()
