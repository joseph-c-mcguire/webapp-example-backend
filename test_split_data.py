import os
import pandas as pd
import unittest
import logging
from split_data import split_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSplitData(unittest.TestCase):

    def setUp(self):
        logger.info("Setting up test data")
        # Create a sample dataset
        self.data = {
            'Feature1': [1, 2, 3, 4, 5],
            'Feature2': [5, 4, 3, 2, 1],
            'Target': [0, 1, 0, 1, 0]
        }
        self.df = pd.DataFrame(self.data)
        self.data_file_path = 'test_predictive_maintenance.csv'
        self.train_val_file_path = 'test_train_val_data.csv'
        self.test_file_path = 'test_test_data.csv'
        self.df.to_csv(self.data_file_path, index=False)

    def tearDown(self):
        logger.info("Tearing down test data")
        # Remove the test files
        if os.path.exists(self.data_file_path):
            os.remove(self.data_file_path)
        if os.path.exists(self.train_val_file_path):
            os.remove(self.train_val_file_path)
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

    def test_split_data(self):
        logger.info("Testing split_data function")
        # Call the split_data function
        split_data(self.data_file_path, self.train_val_file_path,
                   self.test_file_path, test_size=0.4, random_state=42)

        # Check if the files are created
        self.assertTrue(os.path.exists(self.train_val_file_path),
                        f"Training/Validation data file not found at {self.train_val_file_path}")
        self.assertTrue(os.path.exists(self.test_file_path),
                        f"Test data file not found at {self.test_file_path}")

        # Load the split datasets
        train_val_data = pd.read_csv(self.train_val_file_path)
        test_data = pd.read_csv(self.test_file_path)

        # Check the sizes of the split datasets
        self.assertEqual(len(train_val_data), 3,
                         f"Expected 3 rows in training/validation data, got {len(train_val_data)}")
        self.assertEqual(len(test_data), 2,
                         f"Expected 2 rows in test data, got {len(test_data)}")

        # Check if the data is correctly split
        self.assertIn('Feature1', train_val_data.columns,
                      "Feature1 not found in training/validation data columns")
        self.assertIn('Feature2', train_val_data.columns,
                      "Feature2 not found in training/validation data columns")
        self.assertIn('Target', train_val_data.columns,
                      "Target not found in training/validation data columns")
        self.assertIn('Feature1', test_data.columns,
                      "Feature1 not found in test data columns")
        self.assertIn('Feature2', test_data.columns,
                      "Feature2 not found in test data columns")
        self.assertIn('Target', test_data.columns,
                      "Target not found in test data columns")


if __name__ == '__main__':
    unittest.main()
