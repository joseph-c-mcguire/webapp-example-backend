
import os
import pandas as pd
import unittest
from split_data import split_data


class TestSplitData(unittest.TestCase):

    def setUp(self):
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
        # Remove the test files
        if os.path.exists(self.data_file_path):
            os.remove(self.data_file_path)
        if os.path.exists(self.train_val_file_path):
            os.remove(self.train_val_file_path)
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

    def test_split_data(self):
        # Call the split_data function
        split_data(self.data_file_path, self.train_val_file_path,
                   self.test_file_path, test_size=0.4, random_state=42)

        # Check if the files are created
        self.assertTrue(os.path.exists(self.train_val_file_path))
        self.assertTrue(os.path.exists(self.test_file_path))

        # Load the split datasets
        train_val_data = pd.read_csv(self.train_val_file_path)
        test_data = pd.read_csv(self.test_file_path)

        # Check the sizes of the split datasets
        self.assertEqual(len(train_val_data), 3)
        self.assertEqual(len(test_data), 2)

        # Check if the data is correctly split
        self.assertIn('Feature1', train_val_data.columns)
        self.assertIn('Feature2', train_val_data.columns)
        self.assertIn('Target', train_val_data.columns)
        self.assertIn('Feature1', test_data.columns)
        self.assertIn('Feature2', test_data.columns)
        self.assertIn('Target', test_data.columns)


if __name__ == '__main__':
    unittest.main()
