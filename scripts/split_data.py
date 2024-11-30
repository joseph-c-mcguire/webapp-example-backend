import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
from app.config import Config

logger = logging.getLogger(__name__)


def split_data():
    config = Config()
    raw_data_path = config.RAW_DATA_PATH
    _ = config.PROCESSED_DATA_PATH
    test_data_path = config.TEST_DATA_PATH

    logger.info(f"Loading raw data from {raw_data_path}")
    if not raw_data_path.exists():
        logger.error(f"Raw data file not found at {raw_data_path}")
        return

    df = pd.read_csv(raw_data_path)
    logger.info("Splitting data into training/validation and test sets")
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)

    logger.info(f"Saving test data to {test_data_path}")
    test_df.to_csv(test_data_path, index=False)
    logger.info("Test data split completed successfully")


if __name__ == "__main__":
    split_data()
