"""
Run Module

This module initializes and runs the Flask application.

Parameters
----------
None

Returns
-------
None
"""

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
