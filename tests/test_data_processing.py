import unittest
import pandas as pd
import os
from app.utils.data_preprocessing import preprocess_data, split_data


class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        # Create a sample dataframe for testing
        self.data = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [10, 20, 30, 40, 50],
                "col3": ["A", "B", "A", "B", "C"],
            }
        )

    def test_preprocess_data(self):
        # Test preprocessing with dropping, scaling, and encoding
        columns_to_drop = ["col1"]
        columns_to_scale = ["col2"]
        columns_to_encode = ["col3"]

        preprocessor, preprocessed_data = preprocess_data(
            self.data, columns_to_drop, columns_to_scale, columns_to_encode
        )

        self.assertEqual(preprocessed_data.shape[1], 4)  # 1 scaled + 3 encoded columns
        self.assertNotIn("col1", preprocessed_data.columns)

    def test_split_data(self):
        # Create temporary files for testing
        data_file_path = "test_data.csv"
        train_val_file_path = "train_val_data.csv"
        test_file_path = "test_data_split.csv"

        self.data.to_csv(data_file_path, index=False)

        split_data(
            data_file_path,
            train_val_file_path,
            test_file_path,
            test_size=0.2,
            random_state=42,
        )

        self.assertTrue(os.path.exists(train_val_file_path))
        self.assertTrue(os.path.exists(test_file_path))

        train_val_data = pd.read_csv(train_val_file_path)
        test_data = pd.read_csv(test_file_path)

        self.assertEqual(len(train_val_data) + len(test_data), len(self.data))

        # Clean up temporary files
        os.remove(data_file_path)
        os.remove(train_val_file_path)
        os.remove(test_file_path)


if __name__ == "__main__":
    unittest.main()
