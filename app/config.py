import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "models"))
    DATA_PATH = os.getenv("DATA_PATH", os.path.join(BASE_DIR, "data"))
    DEBUG = False
    TESTING = False
    PORT = int(os.getenv("PORT", 80))


class DevelopmentConfig(Config):
    DEBUG = True


class TestingConfig(Config):
    TESTING = True


class ProductionConfig(Config):
    DEBUG = False
