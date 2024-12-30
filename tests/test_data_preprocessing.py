import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import pandas as pd
from src.preprocessing.data_preprocessing import ResumeTextPreprocessor
from src.training.training import call_data, data_preparing_func

class TestDataPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests"""
        cls.raw_data = call_data()
        cls.preprocessed_data = data_preparing_func(cls.raw_data)
        cls.num_classes = len(cls.raw_data['Category'].unique())
    
    def test_data_splitting(self):
        """Test if data is split correctly"""
        # Check if all required keys exist
        required_keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']
        for key in required_keys:
            self.assertIn(key, self.preprocessed_data, f"Missing key: {key}")
    
    def test_data_shapes(self):
        """Test if data shapes are consistent"""
        # Check if X and y have same number of samples for each split
        self.assertEqual(len(self.preprocessed_data['X_train']), len(self.preprocessed_data['y_train']),
                        "Training data and labels have different lengths")
        self.assertEqual(len(self.preprocessed_data['X_val']), len(self.preprocessed_data['y_val']),
                        "Validation data and labels have different lengths")
        self.assertEqual(len(self.preprocessed_data['X_test']), len(self.preprocessed_data['y_test']),
                        "Test data and labels have different lengths")
    
    def test_data_types(self):
        """Test if data types are correct"""
        # Check if X data is numpy array
        self.assertIsInstance(self.preprocessed_data['X_train'], np.ndarray,
                            "X_train is not a numpy array")
        # Check if y data is numpy array
        self.assertIsInstance(self.preprocessed_data['y_train'], np.ndarray,
                            "y_train is not a numpy array")
    
    def test_label_encoding(self):
        """Test if labels are properly encoded"""
        # Check if y data is one-hot encoded
        self.assertEqual(self.preprocessed_data['y_train'].shape[1], self.num_classes,
                        "Labels are not properly one-hot encoded")
        
        # Check if all values are either 0 or 1
        self.assertTrue(np.all(np.logical_or(self.preprocessed_data['y_train'] == 0, 
                                           self.preprocessed_data['y_train'] == 1)),
                       "One-hot encoded values are not binary")
    
    def test_vocabulary_size(self):
        """Test if vocabulary size is within expected range"""
        self.assertIn('vocab_size', self.preprocessed_data,
                     "vocab_size not found in preprocessed data")
        self.assertGreater(self.preprocessed_data['vocab_size'], 0,
                          "Vocabulary size should be greater than 0")
        self.assertLess(self.preprocessed_data['vocab_size'], 100000,
                       "Vocabulary size suspiciously large")
    
    def test_sequence_length(self):
        """Test if all sequences have the same length"""
        expected_length = self.preprocessed_data['X_train'].shape[1]
        self.assertTrue(all(x.shape[1] == expected_length for x in 
                          [self.preprocessed_data['X_train'],
                           self.preprocessed_data['X_val'],
                           self.preprocessed_data['X_test']]),
                       "Sequences have inconsistent lengths")
    
    def test_data_distribution(self):
        """Test if data split ratios are approximately correct"""
        total_samples = (len(self.preprocessed_data['X_train']) +
                        len(self.preprocessed_data['X_val']) +
                        len(self.preprocessed_data['X_test']))
        
        # Test approximate split ratios (allowing for some rounding differences)
        self.assertAlmostEqual(len(self.preprocessed_data['X_train']) / total_samples,
                              0.7, delta=0.05,
                              msg="Training set ratio is not approximately 70%")
        self.assertAlmostEqual(len(self.preprocessed_data['X_val']) / total_samples,
                              0.1, delta=0.05,
                              msg="Validation set ratio is not approximately 10%")
        self.assertAlmostEqual(len(self.preprocessed_data['X_test']) / total_samples,
                              0.2, delta=0.05,
                              msg="Test set ratio is not approximately 20%")
    
    def test_no_data_leakage(self):
        """Test that there's no overlap between train, validation and test sets"""
        train_samples = set(map(tuple, self.preprocessed_data['X_train']))
        val_samples = set(map(tuple, self.preprocessed_data['X_val']))
        test_samples = set(map(tuple, self.preprocessed_data['X_test']))
        
        self.assertEqual(len(train_samples.intersection(val_samples)), 0,
                        "Found overlap between training and validation sets")
        self.assertEqual(len(train_samples.intersection(test_samples)), 0,
                        "Found overlap between training and test sets")
        self.assertEqual(len(val_samples.intersection(test_samples)), 0,
                        "Found overlap between validation and test sets")

if __name__ == '__main__':
    unittest.main(verbosity=1) 