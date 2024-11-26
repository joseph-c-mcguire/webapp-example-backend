from flask_cors import CORS

# Initialize Flask extensions
cors = CORS()


def init_app(app):
    """
    Initialize all Flask extensions with the given app.

    :param app: The Flask application instance
    """
    cors.init_app(app)
