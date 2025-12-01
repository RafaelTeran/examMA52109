## 3) Create a new test file in folder `tests`, called `test_preprocessing.py`.
## In this file, write exactly three unit tests that verify important and meaningful
## aspects of the module preprocessing.py. Your tests must:
##  - be your own design,
##  - include a brief comment above each test explaining what real problem or bug in
##    preprocessing your test is intended to detect,
##  - check behaviours that genuinely matter for the correctness or reliability of the
##    clustering pipeline.
##
## The focus of this task is not on writing long code, but on showing clear 
## understanding of what the preprocessing module is supposed to do, what could go 
## wrong in real usage, and how a well-designed test can detect those issues.
##
## Avoid tests with superficial checks, or without thoughtful explanatory comments.
## [15]

import unittest
import pandas as pd
import numpy as np
from cluster_maker.preprocessing import select_features, standardise_features

class TestPreprocessing(unittest.TestCase):

# ERROR CHECKED: select_features rejects non-numeric columns
    def test_select_features_rejects_non_numeric(self):
        """
        Test that select_features raises an error if non-numeric columns are requested.
        
        Real problem detected: Machine learning algorithms like K-Means calculate 
        mathematical distances. If string/text data slips through preprocessing, 
        the algorithm will crash with a TypeError during calculation. 
        This test ensures the pipeline fails early and clearly.
        """
        # Setup: DataFrame with a numeric column and a string column
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'city': ['London', 'Paris', 'New York']
        })
        
        # Action & Assert: Requesting the 'city' column should trigger a TypeError
        with self.assertRaises(TypeError):
            select_features(df, ['age', 'city'])

# ERROR CHECKED: select_features detects missing/non-existent columns
    def test_select_features_detects_missing_columns(self):
        """
        Test that select_features raises an error if a requested column does not exist.
        
        Real problem detected: Users often make typos in column names (e.g., 'Colour' 
        vs 'Color'). Without this check, the system might silently return an empty 
        DataFrame or ignore the missing column, leading to a model trained on 
        incomplete features without the user knowing.
        """
        # Setup: DataFrame with columns 'A' and 'B'
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        
        # Action & Assert: Requesting non-existent column 'C' should trigger a KeyError
        with self.assertRaises(KeyError):
            select_features(df, ['A', 'C'])

# ERROR CHECKED: standardise_features correctly standardises data
    def test_standardise_features_math_correctness(self):
        """
        Test that standardise_features produces output with mean ~0 and std ~1.
        
        Real problem detected: If the standardization formula is implemented incorrectly 
        (e.g., forgetting to divide by std dev), features with larger magnitudes 
        (like Salary) will dominate features with smaller magnitudes (like Age), 
        resulting in biased and incorrect clusters.
        """
        # Setup: A simple array [10, 20, 30]
        # Mean is 20, Std is approx 8.16
        data = np.array([[10.0], [20.0], [30.0]])
        
        # Action
        scaled_data = standardise_features(data)
        
        # Assert: Check statistical properties
        # We use assertAlmostEqual because floating point math is rarely exactly 0.0
        calculated_mean = np.mean(scaled_data)
        calculated_std = np.std(scaled_data)
        
        self.assertAlmostEqual(calculated_mean, 0.0, places=5, 
                               msg="Scaled mean should be approx 0")
        self.assertAlmostEqual(calculated_std, 1.0, places=5, 
                               msg="Scaled std deviation should be approx 1")

if __name__ == "__main__":
    unittest.main()