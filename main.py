import os
import sys
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel
import joblib
from typing import Optional

import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np

# Updated imports using package structure
from src.utils import validate_and_rename_columns, ExtractTextFromFile
from src.training import Imbalanced_Data_Handler, data_preparing_func, preprocessor_func
from src.model import TextClassifier
from src.inference import ResumePredictor
from src.utils.logger import logger, get_log_status

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Initialize FastAPI
app = FastAPI(
    title="Resume Classification API",
    description="API to classify resumes=",
    version="0.3.0"
)

## Loading the models
try:
    # Register custom model class before loading
    keras.saving.get_custom_objects().clear()
    keras.saving.get_custom_objects()["TextClassifier"] = TextClassifier

    # Load Model
    model = load_model("src/model/best_model.keras")
    logger.info("Model loaded successfully")
except FileNotFoundError:
    logger.error("Model file not found! Please check the path.")
    model = None

# Load the Tokenizer
try:
    with open("src/model/trained_tokenizer.json", "r", encoding='utf-8') as f:
        tokenizer = tokenizer_from_json(f.read())
        logger.info("Tokenizer loaded successfully")
except FileNotFoundError:
    logger.error("Tokenizer file not found! Please try again later.")
    tokenizer = None

# Load the Encoder
try:
    encoder = joblib.load('src/model/OHEncoder.joblib')
    logger.info("Encoder loaded successfully")
except FileNotFoundError:
    logger.error("Encoder file not found! Please try again later.")
    encoder = None

# Load the Preprocessor
try:
    preprocessor = preprocessor_func(train=False, Token=tokenizer, Encoder=encoder)
    logger.info("Preprocessor loaded successfully")
except Exception as e:
    logger.error(f"Failed to load preprocessor: {str(e)}")
    preprocessor = None

# Input Model
class ResumeInput(BaseModel):
    resume_text: str

# Prediction Model
class ClassificationResponse(BaseModel):
    category: str
    confidence: float
    status: str
    request_id: str | None = None
    extracted_text: str | None = None

class RetrainingResponse(BaseModel):
    status: str
    message: str
    progress: str
    task_id: str  # Add this to track the training job

class RetrainingConfig(BaseModel):
    epochs: int = 20
    strategy: str = "weighted"
    # Add other configurable parameters

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
            logger.info(f"Successfully extracted text from PDF: {filename}")
        elif file_extension == ".docx" or content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            extracted_text = text_extractor.docExtractor(file_bytes)
            logger.info(f"Successfully extracted text from DOCX: {filename}")
        elif file_extension == ".doc":
            raise HTTPException(status_code=400, detail=f".doc files are not fully supported yet. Please convert to .docx or .pdf. Error: {str(e) if 'e' in locals() else 'Not implemented'}")
        else:
            logger.warning(f"Unsupported file type for extra: {filename}")
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}. Please upload .pdf, .docx, or .doc files.")
        if not extracted_text.strip():
            logger.warning(f"No text extraxted from the file: {filename}. It might be empty or protected.")
            raise HTTPException(status_code=400, detail=f"Could not extract text from file: {filename}. The file might be empty or password-protected.")

        return extracted_text, filename

    except HTTPException as e: # Re-raise HTTPExceptions
        raise e
    except Exception as e:
        # Log the exception
        logger.error(f"Error during text extraction for {filename}: {e}", exc_info=True)
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
    request_id = os.urandom(8).hex()
    logger.info(f"[{request_id}] Received request for /classify_resume_text/")
    if not model:
        logger.error(f"[{request_id}] Model not available for prediction.")
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")  # Fixed typo
    if not preprocessor:
        logger.error(f"[{request_id}] Preprocessor not available for prediction.")
        raise HTTPException(status_code=504, detail="Preprocessor not loaded. Service unavailable.")

    try:
        text = resume_txt.resume_text

        if isinstance(text, str):
            logger.debug(f"[{request_id}] Text for prediction (first 50 chars): {text[:50]}")

            try:
                # Generate Predictions
                predictor = ResumePredictor(model, preprocessor)
                logger.debug(f"[{request_id}] Successfully created predictor instance")
                
                # Output Classification Category
                result = predictor.predict(text)  # Use the input text
                logger.debug(f"[{request_id}] Successfully generated prediction")
                
                prediction_category = result['category'][0]
                prediction_confidence = result['confidence']
                test = "Success Using Text."
                logger.debug(f"[{request_id}] Successfully extracted prediction results")

                return ClassificationResponse(category=prediction_category, confidence=prediction_confidence, status=test, extracted_text=text)

            except Exception as e:
                logger.error(f"[{request_id}] Error during prediction: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

    except Exception as e:  # Changed to capture any other exceptions
        logger.error(f"[{request_id}] Unhandled exception in /classify_resume_text/ endpoint: {e}", exc_info=True)
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
    
    request_id = os.urandom(8).hex()
    logger.info(f"[{request_id}] Received request for /classify_resume_text/")
    if not model:
        logger.error(f"[{request_id}] Model not available for prediction.")
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")  # Fixed typo
    if not preprocessor:
        logger.error(f"[{request_id}] Preprocessor not available for prediction.")
        raise HTTPException(status_code=504, detail="Preprocessor not loaded. Service unavailable.")

    try:
        logger.debug(f"[{request_id}] Attempting to extract text from uploaded file")
        text, filename = await extract_file_from_file(file)
        logger.debug(f"[{request_id}] Successfully extracted text from file: {filename}")
        
        if isinstance(text, str):
            try:
                # Generate Predictions
                logger.debug(f"[{request_id}] Creating predictor instance")
                predictor = ResumePredictor(model, preprocessor)
                logger.debug(f"[{request_id}] Successfully created predictor instance")
                
                # Output Classification Category
                logger.debug(f"[{request_id}] Generating prediction")
                result = predictor.predict(text)
                logger.debug(f"[{request_id}] Successfully generated prediction")
                
                prediction_category = result['category'][0]
                prediction_confidence = result['confidence']
                test = "Success Using File."
                logger.debug(f"[{request_id}] Successfully extracted prediction results")

                return ClassificationResponse(category=prediction_category, confidence=prediction_confidence, status=test, extracted_text=text)
            except Exception as e:
                logger.error(f"[{request_id}] Error during prediction: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
        else:
            logger.error(f"[{request_id}] Invalid text format received from file extraction")
            raise AssertionError("The data is not of the text format or not correctly extracted!")
            
    except HTTPException as e:
        logger.error(f"[{request_id}] HTTP Exception occurred: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"[{request_id}] Unhandled exception in /classify_resume/file/ endpoint: {e}", exc_info=True)
        print(f"Error during classification: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during classification: {str(e)}")


@app.post("/classify_resume/train/", response_model=RetrainingResponse)
async def modelRetraining(background_tasks: BackgroundTasks):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")
    
    try:
        handler = Imbalanced_Data_Handler(preprocessor_func(), 'weighted')
    except Exception as e:
        raise HTTPException(
            status_code=511, 
            detail="Data Preprocessor Function failed to load properly. Please check the parameters and configuration."
        )

    ## TODO: ERROR! The data_preparing_func is not working properly needs to be fixed.
    try:
        fin_Data = data_preparing_func(preprocessor_func(), call_data(), training=True)
    except Exception as e:
        raise HTTPException(
            status_code=512, 
            detail="Data Preparation Function failed. Please check the data and preprocessing parameters."
        )

    try:
        model_retrained, calls = model_comp(fin_Data, preprocessor_func())
    except Exception as e:
        raise HTTPException(
            status_code=513, 
            detail="Model compilation failed. Please check the model configuration."
        )

    try:
        train_model = train_step(Handler=handler, model=model_retrained, callbacks=calls, data=fin_Data, Epochs=20)
        status = "started"
        message = "Training has started successfully."
        progress = "Training in progress."
    except Exception as e:
        raise HTTPException(
            status_code=514, 
            detail="Model training failed. Please check the training configuration and data."
        )

    try:
        model_retrained.save("src/model/best_model_retrained.keras")
        status = "completed"
        message = "Model saved successfully!"
        progress = "Training completed and model saved."
    except Exception as e:
        raise HTTPException(
            status_code=515, 
            detail="Model saving failed. Please check the file system permissions."
        )

    background_tasks.add_task(train_and_save_model, handler, fin_Data)
    
    return RetrainingResponse(
        status=status,
        message=message,
        progress=progress
    )


@app.get("/")
async def read_root():
    return FileResponse("frontend/index.html")


@app.get("/logs/status")
async def get_log_status_endpoint():
    """Returns information about the current logging setup"""
    return get_log_status()

# Add this after creating the FastAPI app
app.mount("/static", StaticFiles(directory="frontend"), name="static")
