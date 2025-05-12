import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib

import keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json


from src.preprocessing.data_preprocessing import ResumeTextPreprocessor, NLPPreprocessor
from src.utils.helpers import validate_and_rename_columns
from src.training.training import Imbalanced_Data_Handler, data_preparing_func, call_data, preprocessor_func, model_comp
from src.model.model import TextClassifier

class ResumePredictor:
    def __init__(self, model_path: str, preprocessor: NLPPreprocessor = None):
        """
        Initialize the predictor with a trained model and preprocessor
        
        Args:
            model_path (str): Path to the saved model file
            preprocessor (NLPPreprocessor, optional): Preprocessor instance
        """
        # Register custom model class before loading
        keras.saving.get_custom_objects().clear()
        keras.saving.get_custom_objects()["TextClassifier"] = TextClassifier
        
        self.model = load_model(model_path)
        
        if preprocessor is None:
            # Initialize preprocessor and fit tokenizer on training data
            self.preprocessor = preprocessor_func(train=True)
            # Get the training data to fit tokenizer
            dataf = call_data()
            # Fit tokenizer on cleaned text data
            self.preprocessor.tokenizer.fit_on_texts(dataf['cleaned_text'])
        else:
            self.preprocessor = preprocessor
            
        self.resume_cleaner = ResumeTextPreprocessor()
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess the input text"""
        return self.resume_cleaner.process_and_check(text)
    
    def predict(self, text: str) -> dict:
        """
        Make prediction for a single resume text
        
        Args:
            text (str): Raw resume text
            
        Returns:
            dict: Dictionary containing predicted category and confidence score
        """
        # Clean and preprocess the text
        cleaned_text = self.preprocess_text(text)
        
        # Convert text to sequence using preprocessor
        processed_text = self.preprocessor.predict_process(cleaned_text)
        
        # Make prediction
        predictions = self.model.predict(processed_text)
        predicted_class_index = np.argmax(predictions)
        print(predicted_class_index.shape, predicted_class_index.reshape(-1, 1).shape, predicted_class_index.reshape(1, -1).shape)
        confidence_score = float(predictions[0][predicted_class_index])
        
        # Get predicted category name
        one_hot = np.zeros((1, predictions.shape[1]))
        one_hot[0, predicted_class_index] = 1
        predicted_category = self.preprocessor.decode_predictions(one_hot)[0]
        
        return {
            'category': predicted_category,
            'confidence': confidence_score,
            'all_probabilities': {
                category: float(prob) 
                for category, prob in zip(self.preprocessor.onehot_encoder.get_feature_names_out(), predictions[0])
            }
        }
    
    def predict_batch(self, texts: list) -> list:
        """
        Make predictions for multiple resume texts
        
        Args:
            texts (list): List of resume texts
            
        Returns:
            list: List of prediction dictionaries
        """
        predictions = []
        for text in texts:
            pred = self.predict(text)
            predictions.append(pred)
        return predictions



if __name__ == "__main__":
    # Example usage
    MODEL_PATH = r"C:\Projs\COde\ResAnalysis\Resume-Analysis-NLP\best_model.keras"

    # Load trained Tokenizer
    with open("trained_tokenizer.json", "r", encoding='utf-8') as f:
        loaded_tokenizer = tokenizer_from_json(f.read())

    encoder = joblib.load('OHEncoder.joblib')

    # Register custom model class before loading
    keras.saving.get_custom_objects().clear()
    keras.saving.get_custom_objects()["TextClassifier"] = TextClassifier

    model = load_model(MODEL_PATH)
    
    # Initialize preprocessor with vocabulary from training data
    prep = preprocessor_func(train=False, Token=loaded_tokenizer, Encoder=encoder)
    
    # Example resume text
    sample_resume = """
    Experienced software engineer with 5 years of experience in Python development.
    Skilled in machine learning, data analysis, and web development using Django.
    Led multiple projects and mentored junior developers.
    """
    
    # # Preprocess the text
    # cleaned_text = preprocess_text(sample_resume)
    # print("Cleaned text:\n", cleaned_text)
    
    # # Process for prediction
    # processed = prep.predict_process(cleaned_text)
    # print("\n\nProcessed sequence:\n", processed)
    
    # # Make prediction
    # predictions = model.predict(processed)
    # predicted_class_index = np.argmax(predictions)
    
    # print("\n\nPrediction output:")
    # print(f"Predicted class index: {predicted_class_index}")
    # print(f"Probabilities: {predictions[0]}")
    
    # Try creating a full predictor
    predictor = ResumePredictor(MODEL_PATH, prep)
    result = predictor.predict(sample_resume)
    
    print("\n\nPredictor Results:")
    print(f"Predicted Category: {result['category']}")
    print(f"Confidence Score: {result['confidence']:.2%}")
