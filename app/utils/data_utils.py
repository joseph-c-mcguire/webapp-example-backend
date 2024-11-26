import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from yaml import safe_load, YAMLError

import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(file_path: str) -> Dict[str, Any]:
    """
    Load the configuration file from a YAML file.

    Parameters
    ----------
    file_path : str
        Path to the YAML file.

    Returns
    -------
    Dict[str, Any]
        Loaded configuration.

    Examples
    --------
    >>> config = load_config("config.yaml")
    >>> print(config)
    """
    logger.info(f"Loading configuration from {file_path}")
    try:
        with open(file_path, "r") as file:
            config = safe_load(file)
        logger.info("Configuration loaded successfully")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {file_path}")
    except YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    return {}


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Examples
    --------
    >>> data = load_data("data/dataset.csv")
    >>> print(data.head())
    """
    logger.info(f"Loading data from {file_path}")
    data = pd.read_csv(file_path)
    logger.info("Data loaded successfully")
    return data


def preprocess_data(
    data: pd.DataFrame,
    columns_to_drop: List[str] = None,
    columns_to_scale: List[str] = None,
    columns_to_encode: List[str] = None,
) -> Tuple[ColumnTransformer, np.ndarray]:
    """
    Preprocess the dataset by handling missing values, outliers, and normalizing the data.

    Parameters
    ----------
    data : pd.DataFrame
        Raw dataset.
    columns_to_drop : List[str], optional
        List of columns to drop from the dataset, by default None.
    columns_to_scale : List[str], optional
        List of columns to scale/normalize, by default None.
    columns_to_encode : List[str], optional
        List of columns to encode (e.g., categorical columns), by default None.

    Returns
    -------
    Tuple[ColumnTransformer, np.ndarray]
        Preprocessor and preprocessed dataset.

    Examples
    --------
    >>> preprocessor, preprocessed_data = preprocess_data(
    >>>     data,
    >>>     columns_to_drop=["col1"],
    >>>     columns_to_scale=["col2"],
    >>>     columns_to_encode=["col3"]
    >>> )
    >>> print(preprocessed_data.head())
    """
    logger.info("Starting data preprocessing")
    if columns_to_drop is None:
        columns_to_drop = []
    if columns_to_scale is None:
        columns_to_scale = []
    if columns_to_encode is None:
        columns_to_encode = []

    # Create a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("drop_columns", "drop", columns_to_drop),
            ("scale_columns", StandardScaler(), columns_to_scale),
            ("encode_columns", OneHotEncoder(), columns_to_encode),
        ]
    )
    # Fit the column transformer
    scaled_data = preprocessor.fit_transform(data)
    logger.info("Data preprocessing completed")
    return preprocessor, scaled_data


def perform_eda(data: pd.DataFrame) -> None:
    """
    Perform exploratory data analysis on the dataset.

    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed dataset.

    Returns
    -------
    None

    Examples
    --------
    >>> perform_eda(data)
    """
    logger.info("Starting exploratory data analysis (EDA)")
    # Plot histograms for each feature
    data.hist(bins=50, figsize=(20, 15))
    plt.show()

    # Plot heatmap of feature correlations
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
    plt.show()
    logger.info("EDA completed")
