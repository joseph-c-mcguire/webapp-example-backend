from decouple import config
from pathlib import Path
from typing import List, Dict, Any, Type
import yaml
import logging

logger = logging.getLogger(__name__)


class Config:
    """
    Base configuration class. Contains default configuration settings.

    Attributes
    ----------
    BASE_DIR : Path
        Absolute path to the directory containing this file.
    MODEL_PATH : Path
        Path to the directory containing models.
    DATA_PATH : Path
        Path to the directory containing data.
    TRAIN_MODEL_CONFIG : Path
        Path to the train_model.yaml configuration file.
    RAW_DATA_PATH : Path
        Path to the raw data directory.
    PROCESSED_DATA_PATH : Path
        Path to the processed data directory.
    PREPROCESSOR_PATH : Path
        Path to the preprocessor pickle file.
    DEBUG : bool
        Flag to enable/disable debug mode.
    TESTING : bool
        Flag to enable/disable testing mode.
    PORT : int
        Port number for the application.
    MODEL_PARAMETERS : Dict[str, Any]
        Model configuration parameters.
    PARAM_GRIDS : Dict[str, Any]
        Parameter grids for models.
    COLUMNS_TO_DROP : List[str]
        Columns to drop from the dataset.
    COLUMNS_TO_SCALE : List[str]
        Columns to scale in the dataset.
    COLUMNS_TO_ENCODE : List[str]
        Columns to encode in the dataset.
    TARGET_COLUMN : str
        Name of the target column.
    TRAIN_TEST_SPLIT : Dict[str, Any]
        Parameters for train-test split.
    MODEL_DIRECTORY : Path
        Directory to save models.

    Methods
    -------
    __new__(cls, config_path: str = None)
        Creates a new instance of the Config class.
    setup_logging(self)
        Set up centralized logging configuration.
    load_training_model_config(self) -> Dict[str, Any]
        Load model-specific configurations from training_model.yaml.
    get_config_class() -> Type['Config']
        Determine the configuration class based on the environment.
    """

    BASE_DIR: Path = Path(__file__).resolve().parent
    MODEL_PATH: Path = config("MODEL_PATH", default=BASE_DIR.parent / "models")
    DATA_PATH: Path = BASE_DIR.parent / "data"
    TRAIN_MODEL_CONFIG: Path = config(
        "TRAIN_MODEL_CONFIG", default=BASE_DIR.parent / "train_model.yaml"
    )
    RAW_DATA_PATH: Path = DATA_PATH / "raw" / "predictive_maintenance.csv"
    PROCESSED_DATA_PATH: Path = Path(
        config("PROCESSED_DATA_PATH", default=DATA_PATH / "processed")
    )
    PREPROCESSOR_PATH: Path = config(
        "PREPROCESSOR_PATH", default=MODEL_PATH / "preprocessor.pkl"
    )
    DEBUG: bool = config("DEBUG", default=False, cast=bool)
    TESTING: bool = config("TESTING", default=False, cast=bool)
    PORT: int = config("PORT", default=80, cast=int)

    _instance = None

    def __new__(cls, config_path: str = None):
        """
        Create a new instance of Config.

        Parameters
        ----------
        config_path : str, optional
            Path to the configuration file, by default None

        Returns
        -------
        Config
            Instance of the Config class.
        """
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.config_path = config_path or cls._instance.TRAIN_MODEL_CONFIG
            cls._instance.training_model_config = (
                cls._instance.load_training_model_config()
            )
            cls._instance.setup_logging()
        return cls._instance

    def setup_logging(self):
        """
        Set up centralized logging configuration.

        Sets the logging level and format for the application.
        """
        logging_level = logging.INFO  # Can be set via config if needed
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(level=logging_level, format=log_format)
        # Optionally, add file handlers or other handlers here

    def load_training_model_config(self) -> Dict[str, Any]:
        """
        Load model-specific configurations from training_model.yaml.

        Returns
        -------
        Dict[str, Any]
            Model configuration parameters.

        Raises
        ------
        FileNotFoundError
            If the training model config file does not exist.
        yaml.YAMLError
            If there is an error parsing the YAML file.
        """
        training_model_config_path = self.TRAIN_MODEL_CONFIG
        if not training_model_config_path.exists():
            logger.error(
                f"Training model config file not found at {training_model_config_path}"
            )
            raise FileNotFoundError(
                f"Training model config file not found at {training_model_config_path}"
            )
        try:
            with open(training_model_config_path, "r") as file:
                config = yaml.safe_load(file)
                logger.info("Loaded training_model.yaml successfully")
                return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing training_model.yaml: {e}")
            raise e

    # Example of accessing model parameters
    MODEL_PARAMETERS: Dict[str, Any] = property(
        lambda self: self.training_model_config.get("models", {})
    )
    PARAM_GRIDS: Dict[str, Any] = property(
        lambda self: self.training_model_config.get("param_grids", {})
    )

    # Add the following properties to access column configurations from training_model.yaml
    COLUMNS_TO_DROP: List[str] = property(
        lambda self: self.training_model_config.get("columns_to_drop", [])
    )
    COLUMNS_TO_SCALE: List[str] = property(
        lambda self: self.training_model_config.get("columns_to_scale", [])
    )
    COLUMNS_TO_ENCODE: List[str] = property(
        lambda self: self.training_model_config.get("columns_to_encode", [])
    )
    TARGET_COLUMN: str = property(
        lambda self: self.training_model_config.get("target_column", "target")
    )  # Added TARGET_COLUMN
    TRAIN_TEST_SPLIT: Dict[str, Any] = property(
        lambda self: self.training_model_config.get("train_test_split", {})
    )  # Added TRAIN_TEST_SPLIT
    MODEL_DIRECTORY: Path = property(
        lambda self: Path(
            self.training_model_config.get("model_directory", self.MODEL_PATH)
        )
    )  # Added MODEL_DIRECTORY

    @staticmethod
    def get_config_class() -> Type["Config"]:
        """
        Determine the configuration class based on the environment.

        Returns
        -------
        Type[Config]
            Appropriate configuration class based on the environment.
        """
        env = config("ENVIRONMENT", default="development")
        if env == "production":
            return ProductionConfig
        elif env == "testing":
            return TestingConfig
        else:
            return DevelopmentConfig


class DevelopmentConfig(Config):
    """
    Development configuration class. Inherits from Config.

    Attributes
    ----------
    DEBUG : bool
        Flag to enable/disable debug mode. Default is True.
    TESTING : bool
        Flag to enable/disable testing mode. Default is True.
    """

    DEBUG: bool = True
    TESTING: bool = True


class TestingConfig(Config):
    """
    Testing configuration class. Inherits from Config.

    Attributes
    ----------
    TESTING : bool
        Flag to enable/disable testing mode. Default is True.
    """

    TESTING: bool = True


class ProductionConfig(Config):
    """
    Production configuration class. Inherits from Config.

    Attributes
    ----------
    DEBUG : bool
        Flag to enable/disable debug mode. Default is False.
    """

    DEBUG: bool = False
