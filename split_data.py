import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    logger.info(f"Splitting data from {data_file_path}")
    if not os.path.exists(data_file_path):
        logger.error(f"Data file not found at {data_file_path}")
        raise FileNotFoundError(f"Data file not found at {data_file_path}")

    df = pd.read_csv(data_file_path)

    # Split the data into training/validation and testing sets
    train_val_data, test_data = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    # Save the split datasets to separate files
    train_val_data.to_csv(train_val_file_path, index=False)
    test_data.to_csv(test_file_path, index=False)

    logger.info(f"Training/Validation data saved to {train_val_file_path}")
    logger.info(f"Testing data saved to {test_file_path}")


if __name__ == "__main__":
    # Define file paths
    data_file_path = os.path.join(
        os.path.dirname(__file__), "data", "predictive_maintenance.csv"
    )
    train_val_file_path = os.path.join(
        os.path.dirname(__file__), "data", "train_val_data.csv"
    )
    test_file_path = os.path.join(os.path.dirname(__file__), "data", "test_data.csv")

    # Call the split_data function
    split_data(data_file_path, train_val_file_path, test_file_path)
