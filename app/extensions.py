from flask_cors import CORS
from flask import Flask

# Initialize Flask extensions
cors = CORS()


def init_app(app: Flask) -> None:
    """
    Initialize all Flask extensions with the given app.

    Parameters
    ----------
    app : Flask
        The Flask application instance

    Returns
    -------
    None
    """
    # Initialize CORS with the Flask app
    cors.init_app(app)
