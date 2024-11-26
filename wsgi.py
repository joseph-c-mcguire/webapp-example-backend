"""
wsgi.py

WSGI entry point for the web application.

Attributes
----------
app : Flask
    The Flask application instance.
"""

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run()
