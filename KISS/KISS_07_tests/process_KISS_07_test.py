import unittest
import pandas as pd
import os
from script_with_functions import read_files, process_dataframes

class TestScript(unittest.TestCase):
    def setUp(self):
        # Create some test TSV files
        self.directory = 'test_directory'
        os.mkdir(self.directory)
        self.test_data = pd.DataFrame({'serving_cell_id': [1, 2, 3], 'power': [10, 20, 30]})
        self.test_data.to_csv(os.path.join(self.directory, 'KISS_07_change_random_cell_power_logfile_1.tsv'), sep='\t', index=False)
        self.test_data.to_csv(os.path.join(self.directory, 'KISS_07_change_random_cell_power_logfile_2.tsv'), sep='\t', index=False)

    def test_read_files(self):
        # Test that read_files reads in the correct number of files
        files = read_files(self.directory)
        self.assertEqual(len(files), 2)

        # Test that the data in the files was read in correctly
        for df in files:
            self.assertTrue(pd.DataFrame.equals(df, self.test_data))

    def test_process_dataframes(self):
        # Test that process_dataframes correctly aggregates the data
        files = read_files(self.directory)
        master_df = process_dataframes(files)
        expected_result = pd.DataFrame({'power': [60]}, index=[1, 2, 3])
        self.assertTrue(pd.DataFrame.equals(master_df, expected_result))

    def tearDown(self):
        # Delete the test directory and files
        for filename in os.listdir(self.directory):
            os.remove(os.path.join(self.directory, filename))
        os.rmdir(self.directory)

if __name__ == '__main__':
    unittest.main()
