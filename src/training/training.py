import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from typing import Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers

from src.preprocessing.data_preprocessing import (
    ResumeTextPreprocessor,
    NLPPreprocessor,
    ImbalancedNLPHandler,
)

from src.model.model import TextClassifier

from src.utils.helpers import validate_and_rename_columns, PlotMetrics


# Calling base dataset.
def call_data():
    # dataf = pd.read_csv(r'C:\Projs\COde\ResAnalysis\Resume-Analysis-NLP\dataset\resume_dataset.csv')
    dataf_new = pd.read_csv(r'C:\Projs\COde\ResAnalysis\Resume-Analysis-NLP\dataset\resume_new.csv')
    dataf = validate_and_rename_columns(dataf_new)
    dataf['cleaned_text'] = dataf['Resume'].apply(ResumeTextPreprocessor().process_and_check)
    return dataf

def preprocessor_func():
    preprocessor = NLPPreprocessor(
        max_words=10000,
        max_length=500,
        embedding_dim=100,
        TFDataset=True
    )
    return preprocessor

def data_preparing_func(preprocessor, dataf: pd.DataFrame) -> pd.DataFrame:
    data = preprocessor.prepare_data(
        texts=dataf['cleaned_text'],
        labels=np.array(dataf['Category']).reshape(-1, 1),
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

def model_comp(data, preprocessor):
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
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    # Initialize Model
    model_2 = TextClassifier(
        vocab_size=vocab_size,
        embed_dim=preprocessor.embedding_dim,
        num_classes=data['num_classes'],
        embedding_matrix=data['embedding_matrix']
    )

    # Compile model
    model_2.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='AUC'),
        ]
    )
        
    return model_2, [early_stopping, checkpoint]


def train_step(Handler, model, callbacks, data):
    # Train model with class weights if using weighted strategy
    if Handler.strategy == "weighted":
        print("Using Weighted Class Balancing!")
        history = model.fit(
            data['train_dataset'],
            validation_data=data['val_dataset'],
            epochs=50,
            class_weight=Handler.calculate_class_weights(data['y_train']), ### To be used when large enough Dataset ###
            callbacks = callbacks
        )
        return history
    elif Handler.strategy == "oversample":
        print("Using Oversampling Class Balancing!")
        history = model.fit(
            data['train_dataset'],
            validation_data=data['val_dataset'],
            epochs=50,
            callbacks=callbacks
        )
        return history
    else:
        history = model.fit(
            data['train_dataset'],
            validation_data=data['val_dataset'],
            epochs=50,
            callbacks=callbacks
        )
        return history


if __name__ == "__main__":
    handler = Imbalanced_Data_Handler(preprocessor_func(), 'weighted')

    fin_Data = data_preparing_func(preprocessor_func(), call_data())

    model, calls = model_comp(fin_Data, preprocessor_func())

    train_model = train_step(Handler=handler, model=model, callbacks=calls, data=fin_Data)
    
    # Plotting the Metrics.
    plotter = PlotMetrics()
    plotter.plot_accuracy(history=train_model)
    plotter.plot_loss(history=train_model)
    plotter.plot_precision(history=train_model)
    plotter.plot_recall(history=train_model)
    plotter.plot_auc(history=train_model)
