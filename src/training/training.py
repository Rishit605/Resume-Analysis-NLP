import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re

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


from src.preprocessing.data_preprocessing import ResumeTextPreprocessor, NLPPreprocessor
from src.model.model import TextAnalysisModel2
from src.utils.helpers import *

# Calling base dataset.
def call_data():
    return pd.read_csv(r'C:\Projs\COde\ResAnalysis\Resume-Analysis-NLP\dataset\resume_dataset.csv')

# Output Lenght
def output_classes(df: pd.DataFrame) -> int:
    
    # Defining the Classes
    classes = list(df['Category'].unique())
    return len(classes)

# Data Preprocessing
def data_preparing_func(dataf: pd.DataFrame):
    data = ResumeTextPreprocessor().prepare_data(
        df=dataf,
        text_column='Resume',
        label_column='Category',
        test_size=0.2,
        val_size=0.1
    )

    return data

def create_and_compile_model(
    vocab_size,
    output,
    embed_dim=100,
    max_length=100,
    conv_units=64,
    kernels=3,
    dense_units=64,
    regularizer=0.01,
    dropout=0.5
):
    model = TextAnalysisModel2(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_length=max_length,
        conv_units=conv_units,
        kernels=kernels,
        dense_units=dense_units,
        output=output,
        regularizer=regularizer,
        dropout=dropout
    )
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',  # Use 'sparse_categorical_crossentropy' if labels are not one-hot encoded
        metrics=['categorical_accuracy']
    )
    
    return model


#  Initialize the model with your parameters
data = data_preparing_func(call_data())
model = create_and_compile_model(
    vocab_size=data['vocab_size'],
    output=data['num_classes'],
    embed_dim=100,
    max_length=100,
    conv_units=64,
    kernels=3,
    dense_units=32,
    regularizer=0.01,
    dropout=0.5
)

def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=10):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )
    return history



if __name__ == "__main__":

    # data = data_preparing_func(ResumeTextPreprocessor().process_and_check(call_data()['Resume'][0]))

    #     # Initialize preprocessor
    # preprocessor = NLPPreprocessor(max_words=10000, max_length=500, embedding_dim=100)

    # # Prepare your cleaned data
    # data = preprocessor.prepare_data(
    #     texts=df['cleaned_resume'],
    #     labels=df['category'],
    #     use_word2vec=True  # Set to True if you want to use Word2Vec embeddings
    # )

    # # Create and compile your model
    # model = create_model(
    #     vocab_size=data['vocab_size'],
    #     embedding_dim=100,
    #     num_classes=data['num_classes'],
    #     embedding_matrix=data['embedding_matrix']  # Pass this if using Word2Vec
    # )

    # # Train the model using the TF datasets
    # history = model.fit(
    #     data['train_dataset'],
    #     validation_data=data['val_dataset'],
    #     epochs=10
    # )

    # print(data['num_classes'])
    print(data)

    # history = train_model(model, data['X_train'], data['y_train'], data['X_val'], data['y_val'], batch_size=32, epochs=200)
    # plot_accuracy(data)
