"""
This module initializes and runs the Flask application.

It creates the Flask app using the application factory pattern, applies Cross-Origin Resource Sharing (CORS)
settings, and runs the app when executed as the main program.
"""

from app import create_app  # Import the application factory function
from flask_cors import CORS  # Import the Flask-CORS extension

# Create the Flask application instance
app = create_app()

# Enable CORS for the app, allowing requests from 'http://localhost' to '/api/*' endpoints
CORS(app, resources={r"/api/*": {"origins": "http://localhost"}})

if __name__ == "__main__":
    # Run the Flask development server
    app.run()
