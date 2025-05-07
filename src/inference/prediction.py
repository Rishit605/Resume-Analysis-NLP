import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json

from src.preprocessing.data_preprocessing import ResumeTextPreprocessor, NLPPreprocessor
from src.utils.helpers import validate_and_rename_columns
from src.training.training import Imbalanced_Data_Handler, data_preparing_func, call_data, preprocessor_func, model_comp

class ResumePredictor:
    def __init__(self, model_arc_path: str, weights_path: str, preprocessor: NLPPreprocessor = None):
        """
        Initialize the predictor with a trained model and preprocessor
        
        Args:
            model_path (str): Path to the saved model file
            preprocessor (NLPPreprocessor, optional): Preprocessor instance
        """
        # with open(model_arc_path, 'r') as json_file:
        #     model_json = json_file.read()
        # self.model = model_from_json(model_json)
        # self.model.load_weights(weights_path)

        # self.model.compile(
        #     optimizer='adam',
        #     loss='categorical_crossentropy',
        #     metrics=[
        #         'accuracy',
        #         tf.keras.metrics.Precision(name='precision'),
        #         tf.keras.metrics.Recall(name='recall'),
        #         tf.keras.metrics.AUC(name='AUC'),
        #     ]
        # )

        self.model = load_model(r"C:\Projs\COde\ResAnalysis\Resume-Analysis-NLP\best_model.keras")
        self.preprocessor = preprocessor_func()
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
        # print(len(processed_text))
        # # Make prediction
        predictions = self.model.predict(processed_text)
        predicted_class_index = np.argmax(predictions)
        confidence_score = float(predictions[0][predicted_class_index])
        # print(predicted_class_index)
        # # Get predicted category name
        # predicted_category = self.preprocessor.decode_predictions([predicted_class_index])[0]
        
        # return {
        #     'category': predicted_category,
        #     'confidence': confidence_score,
        #     'all_probabilities': {
        #         category: float(prob) 
        #         for category, prob in zip(self.preprocessor.label_encoder.classes_, predictions[0])
        #     }
        # }
    
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


def load_predictor(model_path: str, preprocessor_path: str = None) -> ResumePredictor:
    """
    Helper function to load the predictor with saved model and preprocessor
    
    Args:
        model_path (str): Path to saved model
        preprocessor_path (str, optional): Path to saved preprocessor
        
    Returns:
        ResumePredictor: Initialized predictor instance
    """
    # Load preprocessor if path provided
    preprocessor = None
    if preprocessor_path:
        # Add logic to load saved preprocessor state
        pass
        
    return ResumePredictor(model_path, preprocessor)


if __name__ == "__main__":
    # Example usage
    MODEL_JSON_PATH = r"C:\Projs\COde\ResAnalysis\Resume-Analysis-NLP\model_architecture.json"
    WEIGHTS_PATH = "C:/Projs/COde/ResAnalysis/Resume/Analysis-NLP/best_weights.weights.h5"
    
    
    # Initialize predictor
    predictor = ResumePredictor(MODEL_JSON_PATH, WEIGHTS_PATH)
    
    # Example resume text
    sample_resume = """
    Experienced software engineer with 5 years of experience in Python development.
    Skilled in machine learning, data analysis, and web development using Django.
    Led multiple projects and mentored junior developers.
    """
    
    # # Make prediction
    result = predictor.predict(sample_resume)
    print("\nPrediction Results:")
    print(f"Predicted Category: {result['category']}")
    print(f"Confidence Score: {result['confidence']:.2%}")
    # print("\nAll Category Probabilities:")
    # for category, prob in result['all_probabilities'].items():
    #     print(f"{category}: {prob:.2%}")
