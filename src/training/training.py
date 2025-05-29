import os, sys

import re
from typing import Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers

# Updated imports using package structure
from src.preprocessing import (
    ResumeTextPreprocessor,
    NLPPreprocessor,
    ImbalancedNLPHandler,
)
from src.model import TextClassifier
from src.utils import validate_and_rename_columns, PlotMetrics

# Calling base dataset.
def call_data():
    # dataf = pd.read_csv(r'C:\Projs\COde\ResAnalysis\Resume-Analysis-NLP\dataset\resume_dataset.csv')
    dataf_new = pd.read_csv(r'C:\Projs\COde\ResAnalysis\Resume-Analysis-NLP\dataset\resume_new.csv')
    dataf = validate_and_rename_columns(dataf_new)
    dataf['cleaned_text'] = dataf['Resume'].apply(ResumeTextPreprocessor().process_and_check)
    return dataf

def preprocessor_func(train: bool = True, Token = None, Encoder = None):
    if train:
        preprocessor = NLPPreprocessor(
            max_words=10000,
            max_length=500,
            embedding_dim=100,
            TFDataset=True,
            training=train,
        )
        return preprocessor
    else:
        if Token is None:
            raise Exception("Tokenizer object not found! Please write the correct path or tokenizer name!")
            return None
        else:
            preprocessor = NLPPreprocessor(
                max_words=10000,
                max_length=500,
                embedding_dim=100,
                TFDataset=True,
                training=train,
                Token=Token,
                Encoder=Encoder
            )
            return preprocessor

def data_preparing_func(preprocessor, dataf: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    if training:
        data = preprocessor.prepare_data(
            texts=dataf['cleaned_text'],
            labels=np.array(dataf['Category']).reshape(-1, 1),
            use_word2vec=True  # Use Word2Vec embeddings
        )
        return data
    else:
        data = preprocessor.prepare_data(
            texts=dataf['cleaned_text'],
            labels=np.array(dataf['Category']).reshape(1, -1),
            use_word2vec=True  # Use Word2Vec embeddings
        )
        return data

def Imbalanced_Data_Handler(preprocessor, strategy: str = "undersample"):
    """
    Three methods for handling imbalancing:
    
    - Weighted
    - Oversample
    - Undersample

    """
    handler = ImbalancedNLPHandler(
        preprocessor=preprocessor,
        strategy=strategy
    )
    return handler

def class_distribution(data) -> Dict:
    """
    Check initial class distribution
    """
    return handler.get_class_distribution(data["Category"])

def prep_model_data(data, handler, word2vec: bool = True):
    # Prepare data with imbalance handling
    return handler.prepare_balanced_data(data["cleaned_text"], data["Category"], use_word2vec=word2vec)

def class_weights(data):
    class_weights = handler.calculate_class_weights(data["y_train"])
    weights = [class_weights[label] for label in data['y_train']]
    return weights

def model_comp(data, preprocessor, training: bool = True):
    # Setting the Vocab Size from the Embedding Matrix
    if data['embedding_matrix'] is not None:
        vocab_size = data['embedding_matrix'].shape[0] 
    else:
        vocab_size = data['vocab_size']

    # Initialize callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        r'src\model\best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )

    # Initialize Model
    model = TextClassifier(
        vocab_size=vocab_size,
        embed_dim=preprocessor.embedding_dim,
        num_classes=data['num_classes'],
        embedding_matrix=data['embedding_matrix']
    )

    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='AUC'),
        ]
    )

    if training:
        # Clear any custom objects registration to avoid conflicts
        keras.saving.get_custom_objects().clear()
        # Register our custom model class
        keras.saving.get_custom_objects()["TextClassifier"] = TextClassifier
        
        return model, [early_stopping, checkpoint]
    else:
        return model

def train_step(Handler, model, callbacks, data, Epochs):
    # Train model with class weights if using weighted strategy
    if Handler.strategy == "weighted":
        print("Using Weighted Class Balancing!")
        history = model.fit(
            data['train_dataset'],
            validation_data=data['val_dataset'],
            epochs=Epochs,
            class_weight=Handler.calculate_class_weights(data['y_train']), ### To be used when large enough Dataset ###
            callbacks = callbacks
        )
        return history
    elif Handler.strategy == "oversample":
        print("Using Oversampling Class Balancing!")
        history = model.fit(
            data['train_dataset'],
            validation_data=data['val_dataset'],
            epochs=Epochs,
            callbacks=callbacks
        )
        return history
    else:
        history = model.fit(
            data['train_dataset'],
            validation_data=data['val_dataset'],
            epochs=Epochs,
            callbacks=callbacks
        )
        return history


if __name__ == "__main__":
    # Register the custom model class for saving/loading
    keras.saving.get_custom_objects().clear()
    keras.saving.get_custom_objects()["TextClassifier"] = TextClassifier
    
    handler = Imbalanced_Data_Handler(preprocessor_func(), 'weighted')

    fin_Data = data_preparing_func(preprocessor_func(), call_data(), training=True)
    
    model, calls = model_comp(fin_Data, preprocessor_func())

    train_model = train_step(Handler=handler, model=model, callbacks=calls, data=fin_Data, Epochs=20)
    
    # Save the model
    model.save("best_model.keras")

    # Plotting the Metrics.
    plotter = PlotMetrics()
    plotter.plot_combined_metrics(history=train_model)


