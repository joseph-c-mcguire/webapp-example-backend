import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = logging.getLogger(__name__)


def preprocess_data(
    data, columns_to_drop=None, columns_to_scale=None, columns_to_encode=None
):
    """
    Preprocess the dataset: handle missing values, outliers, and normalize the data.

    Parameters:
    data (pd.DataFrame): Raw dataset.
    columns_to_drop (list[str], optional): List of columns to drop from the dataset. Defaults to None.
    columns_to_scale (list[str], optional): List of columns to scale/normalize. Defaults to None.
    columns_to_encode (list[str], optional): List of columns to encode (e.g., categorical columns). Defaults to None.

    Returns:
    Tuple[ColumnTransformer, pd.DataFrame]: Preprocessor and preprocessed dataset.

    Example:
    >>> preprocessor, preprocessed_data = preprocess_data(data, columns_to_drop=["col1"], columns_to_scale=["col2"], columns_to_encode=["col3"])
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


def split_data(
    data_file_path, train_val_file_path, test_file_path, test_size=0.2, random_state=42
):
    """
    Split the data into training/validation and testing sets and save them to separate files.

    Parameters:
    - data_file_path: str, path to the original data file
    - train_val_file_path: str, path to save the training/validation data
    - test_file_path: str, path to save the testing data
    - test_size: float, proportion of the dataset to include in the test split
    - random_state: int, random seed for reproducibility
    """
    logger.info(
        f"Splitting data from {data_file_path} into train/validation and test sets"
    )
    logger.debug(f"Parameters - test_size: {test_size}, random_state: {random_state}")
    if not os.path.exists(data_file_path):
        logger.error(f"Data file not found at {data_file_path}")
        raise FileNotFoundError(f"Data file not found at {data_file_path}")

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
