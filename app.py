from app import create_app
from flask_cors import CORS

app = create_app()
CORS(app, resources={r"/api/*": {"origins": "http://localhost"}})

if __name__ == "__main__":
    app.run()
