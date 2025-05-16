import os
import sys

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import joblib
from typing import Optional

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np

# Updated imports using package structure
from src.utils import validate_and_rename_columns, ExtractTextFromFile
from src.training import Imbalanced_Data_Handler, data_preparing_func, preprocessor_func
from src.model import TextClassifier
from src.inference import ResumePredictor


# Initalize FastAPI
app = FastAPI(
    title="Resume Classification API",
    description="API to classify resumes=",
    version="0.1.0"
)

## Loading the models
try:
    # Load Model
    model = load_model("best_model.keras")
    print("Model loaded succesfully")
except FileNotFoundError:
        print("ERROR: Model file not found!")
        model = None

# Load the Tokenizer
try:
    with open("trained_tokenizer.json", "r", encoding='utf-8') as f:
        tokenizer = tokenizer_from_json(f.read())
        print("Tokenizer loaded successfully.")
except FileNotFoundError:
    print("ERROR: Tokenizer file not found!")
    tokenizer = None

# Load the Encoder
try:
    encoder = joblib.load('OHEncoder.joblib')
    print("Encoder file loaded successfully")
except FileNotFoundError:
    print("ERROR: Encoder file not found!")
    encoder = None

# Load the Preprocessor
try:
    preprocessor = preprocessor_func(train=False, Token=tokenizer, Encoder=encoder)  # Fixed variable name
    print("Preprocessor loaded successfully") 
except Exception as e:
    print("ERROR: Failed to load preprocessor!")
    preprocessor = None

# Input Model
class ResumeInput(BaseModel):
    resume_text: str

# Prediction Model
class ClassificationResponse(BaseModel):
    category: str
    confidence: float
    status: str

# Helper FUnction for Extracting Text
async def extract_file_from_file(file: UploadFile):
    filename = file.filename
    content_type = file.content_type
    file_extension = os.path.splitext(filename)[1].lower()

    file_bytes = await file.read()

    # Initalizing the Text Extractor
    text_extractor = ExtractTextFromFile(file_bytes)

    try:
        if file_extension == ".pdf" or content_type == "application/pdf":
            extracted_text = text_extractor.pdfExtractor(file_bytes)
        elif file_extension == ".docx" or content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            extracted_text = text_extractor.docExtractor(file_bytes)
        elif file_extension == ".doc":
            raise HTTPException(status_code=400, detail=f".doc files are not fully supported yet. Please convert to .docx or .pdf. Error: {str(e) if 'e' in locals() else 'Not implemented'}")

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}. Please upload .pdf, .docx, or .doc files.")
        
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail=f"Could not extract text from file: {filename}. The file might be empty or password-protected.")

        return extracted_text, filename

    except HTTPException as e: # Re-raise HTTPExceptions
        raise e
    except Exception as e:
        # Log the exception
        print(f"Error during text extraction for {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file {filename}: {str(e)}")
    finally:
        await file.close()


# Create API Endpoints
@app.post("/classify_resume/text/", response_model=ClassificationResponse)
async def classifyResumeEP(resume_txt: ResumeInput, typeChecker: bool = True):

    """
    Classifies the provides resume text.

    Arguments: resume_text: takes input as text
    """

    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")  # Fixed typo
    if not preprocessor:
        raise HTTPException(status_code=504, detail="Preprocessor not loaded. Service unavailable.")

    try:
        text = resume_txt.resume_text

        # Generate Predictions
        predictor = ResumePredictor(model, preprocessor)  # Fixed variable name
        
        # Output Classification Category
        result = predictor.predict(text)  # Use the input text
        
        prediction_category = result['category'][0]
        prediction_confidence = result['confidence']
        test = "Success Using Text."

        return ClassificationResponse(category=prediction_category, confidence=prediction_confidence, status=test)

    except HTTPException as e:
        raise e
    except Exception as e:  # Changed to capture any other exceptions
        print(f"Error during classifiction: {e}")
        raise HTTPException(status_code=500, detail=f"An error occured during classifcation: {str(e)}")  # Fixed typo


@app.post("/classify_resume/file/", response_model=ClassificationResponse)
async def classifyResumeEP(file: UploadFile = File(...)):

    """
    Classifies resume from an uploaded file (.pdf, .docx).
    Limited support for .doc (may require 'textract' and system dependencies).

    Arguments: file: takes input as a file/document.

    Return: Category and model's confidence in the category.
    """

    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")  # Fixed typo
    if not preprocessor:
        raise HTTPException(status_code=504, detail="Preprocessor not loaded. Service unavailable.")

    try:
        text, filename = await extract_file_from_file(file)
        
        if isinstance(text, str):
            # Generate Predictions
            predictor = ResumePredictor(model, preprocessor)  # Fixed variable name
            
            # Output Classification Category
            result = predictor.predict(text)  # Use the input text
            
            prediction_category = result['category'][0]
            prediction_confidence = result['confidence']
            test = "Success Using File."

            return ClassificationResponse(category=prediction_category, confidence=prediction_confidence, status=test)
        else:
            raise AssertionError("The data is not of the text format or not correctly extracted!")
            
    except HTTPException as e:
        raise e
    except Exception as e:  # Changed to capture any other exceptions
        print(f"Error during classifiction: {e}")
        raise HTTPException(status_code=500, detail=f"An error occured during classifcation: {str(e)}")  # Fixed typo

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Resume Classification API!"}  # Fixed missing quote

