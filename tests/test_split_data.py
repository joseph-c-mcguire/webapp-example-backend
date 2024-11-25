import os
import pandas as pd
import unittest
from app.data_processing.split_data import split_data
import random


class TestSplitData(unittest.TestCase):

    def setUp(self):
        # Create a sample dataset
        self.data = {
            "Type": [random.choice(["M", "L", "H"]) for _ in range(100)],
            "Air temperature [K]": [random.uniform(290, 310) for _ in range(100)],
            "Process temperature [K]": [random.uniform(300, 320) for _ in range(100)],
            "Rotational speed [rpm]": [random.uniform(1000, 2000) for _ in range(100)],
            "Torque [Nm]": [random.uniform(20, 50) for _ in range(100)],
            "Tool wear [min]": [random.uniform(0, 300) for _ in range(100)],
            "Target": [random.randint(0, 1) for _ in range(100)],
        }
        self.df = pd.DataFrame(self.data)
        self.data_file_path = "test_predictive_maintenance.csv"
        self.train_val_file_path = "test_train_val_data.csv"
        self.test_file_path = "test_test_data.csv"
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
        split_data(
            self.data_file_path,
            self.train_val_file_path,
            self.test_file_path,
            test_size=0.4,
            random_state=42,
        )

        # Check if the files are created
        self.assertTrue(os.path.exists(self.train_val_file_path))
        self.assertTrue(os.path.exists(self.test_file_path))

        # Load the split datasets
        train_val_data = pd.read_csv(self.train_val_file_path)
        test_data = pd.read_csv(self.test_file_path)

        # Check the sizes of the split datasets
        self.assertEqual(len(train_val_data), 60)
        self.assertEqual(len(test_data), 40)

        # Check if the data is correctly split
        self.assertIn("Type", train_val_data.columns)
        self.assertIn("Air temperature [K]", train_val_data.columns)
        self.assertIn("Process temperature [K]", train_val_data.columns)
        self.assertIn("Rotational speed [rpm]", train_val_data.columns)
        self.assertIn("Torque [Nm]", train_val_data.columns)
        self.assertIn("Tool wear [min]", train_val_data.columns)
        self.assertIn("Target", train_val_data.columns)
        self.assertIn("Type", test_data.columns)
        self.assertIn("Air temperature [K]", test_data.columns)
        self.assertIn("Process temperature [K]", test_data.columns)
        self.assertIn("Rotational speed [rpm]", test_data.columns)
        self.assertIn("Torque [Nm]", test_data.columns)
        self.assertIn("Tool wear [min]", test_data.columns)
        self.assertIn("Target", test_data.columns)

    def test_split_data_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            split_data(
                "non_existent_file.csv",
                self.train_val_file_path,
                self.test_file_path,
                test_size=0.4,
                random_state=42,
            )


if __name__ == "__main__":
    unittest.main()
