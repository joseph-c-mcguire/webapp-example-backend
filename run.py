"""
Run Module

This module initializes and runs the Flask application.

Attributes
----------
app : Flask
    The Flask application instance.

Methods
-------
None
"""

from app import create_app

# Create the Flask application instance
app = create_app()

if __name__ == "__main__":
    # Run the Flask application in debug mode
    app.run(debug=True)
