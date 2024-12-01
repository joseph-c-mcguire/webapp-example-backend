import os
import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


def preprocess_data(
    data: pd.DataFrame,
    columns_to_drop: Optional[List[str]] = None,
    columns_to_scale: Optional[List[str]] = None,
    columns_to_encode: Optional[List[str]] = None,
) -> Tuple[ColumnTransformer, pd.DataFrame]:
    """
    Preprocess the dataset: handle missing values, outliers, and normalize the data.

    Parameters
    ----------
    data : pd.DataFrame
        Raw dataset.
    columns_to_drop : list of str, optional
        List of columns to drop from the dataset. Defaults to None.
    columns_to_scale : list of str, optional
        List of columns to scale/normalize. Defaults to None.
    columns_to_encode : list of str, optional
        List of columns to encode (e.g., categorical columns). Defaults to None.

    Returns
    -------
    tuple
        A tuple containing the ColumnTransformer and the preprocessed DataFrame.

    Examples
    --------
    >>> preprocessor, preprocessed_data = preprocess_data(data, columns_to_drop=["col1"], columns_to_scale=["col2"], columns_to_encode=["col3"])
    >>> print(preprocessed_data.head())
    """
    logger.info("Starting data preprocessing")

    # Initialize lists if None
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

    # Fit the column transformer and transform the data
    transformed_data = preprocessor.fit_transform(data)

    # Get the new column names after transformation
    new_columns = []
    for name, transformer, columns in preprocessor.transformers:
        if name == "drop_columns":
            continue
        if name == "scale_columns":
            new_columns.extend(columns)
        if name == "encode_columns":
            encoder = preprocessor.named_transformers_["encode_columns"]
            encoded_columns = encoder.get_feature_names_out(columns)
            new_columns.extend(encoded_columns)

    # Create the preprocessed DataFrame with the new columns
    preprocessed_data = pd.DataFrame(transformed_data, columns=new_columns)

    logger.info("Data preprocessing completed")

    return preprocessor, preprocessed_data


def split_data(
    data_file_path: str,
    train_val_file_path: str,
    test_file_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Split the data into training/validation and testing sets and save them to separate files.

    Parameters
    ----------
    data_file_path : str
        Path to the original data file.
    train_val_file_path : str
        Path to save the training/validation data.
    test_file_path : str
        Path to save the testing data.
    test_size : float, optional
        Proportion of the dataset to include in the test split. Defaults to 0.2.
    random_state : int, optional
        Random seed for reproducibility. Defaults to 42.

    Raises
    ------
    FileNotFoundError
        If the data file is not found at the specified path.

    Examples
    --------
    >>> split_data("data.csv", "train_val.csv", "test.csv", test_size=0.2, random_state=42)
    """
    logger.info(
        f"Splitting data from {data_file_path} into train/validation and test sets"
    )
    logger.debug(f"Parameters - test_size: {test_size}, random_state: {random_state}")

    # Check if the data file exists
    if not os.path.exists(data_file_path):
        logger.error(f"Data file not found at {data_file_path}")
        raise FileNotFoundError(f"Data file not found at {data_file_path}")

    # Load the data
    df = pd.read_csv(data_file_path)
    logger.debug(f"Data loaded successfully with shape: {df.shape}")

    # Split the data into training/validation and testing sets
    train_val_data, test_data = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    logger.debug(f"Train/validation data shape: {train_val_data.shape}")
    logger.debug(f"Test data shape: {test_data.shape}")

    # Save the split datasets to separate files
    train_val_data.to_csv(train_val_file_path, index=False)
    test_data.to_csv(test_file_path, index=False)

    logger.info(f"Training/Validation data saved to {train_val_file_path}")
    logger.info(f"Testing data saved to {test_file_path}")
