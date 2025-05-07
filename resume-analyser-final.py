import matplotlib.pyplot as plt
import re

import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

# Importing the Dataset
df = pd.read_csv('/content/dataset/resume_dataset.csv')
print(df.head(10))

df2 = df.copy(deep=True)
print(df2.head(10))

# Defining the Classes
classes = list(df['Category'].unique())
print(classes, len(classes))

class_idx = {key: value for value, key in enumerate(classes)}
print(classes)

# DATA CLEANING

import string


def rem_punc(s):
    punc = string.punctuation
    return [i for i in s if i not in punc]


def rem_sw(s):
    sw = set(stopwords)
    return [i for i in s if i not in sw]


def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)
    # Remove punctuations
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    # Convert to lowercase
    text = text.lower()
    return text


print(df2['Resume'])

df2['normalized_text'] = df2['Resume'].apply(preprocess_text)

length = df['Resume'].shape
length

# Decoding the text
j = 0
i = 0
l = []
for i in range(length[0]):
    try:
        df2['Resume'][i] = eval(df2['Resume'][i]).decode()
    except:
        l.append(i)
        pass

# Making a new copy of the dataframe to not alter the original
df2["res_new"] = df['Resume']
df2 = df2.drop(l, axis=0)
# print(df[30:40])
df3 = df2.reset_index(drop=True)

df3['normalized_text'] = df3['Resume'].apply(preprocess_text)

df4 = df3[['Category', 'normalized_text']]
print(df4.head(10))

# Encoding the Label
## initialize a LabelEncoder object
label_encoder = LabelEncoder()

# encode the categorical labels as numerical values
df4['Label'] = label_encoder.fit_transform(df4['Category'])

print(df4.head(10))

### PARAMETERS ###
vocab_size = 10000
embed_dim = 64
max_length = 120
trunc_type = 'pre'
oov_tok = ""

# TOKENIZATION
token = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
token.fit_on_texts(df4['normalized_text'].values)

word_idx = token.word_index

## USING ONLY TRAINING SEQUENCE AS THE DATASET IS SAMLL
train_seq = token.texts_to_sequences(df4['normalized_text'].values)
train_padded = pad_sequences(train_seq, maxlen=max_length, truncating=trunc_type)

## MODEL BUILDING
model = Sequential([
    layers.Embedding(vocab_size, embed_dim, input_length=max_length),
    layers.Conv1D(128, 8, activation='relu'),
    layers.GlobalMaxPooling1D(),

    layers.Dense(48, activation='relu'),
    layers.Dense(25, activation='softmax'),
])

model.summary()

model.output_shape

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the compiled model
y_train = pd.get_dummies(df4['label']).values
# [ #train_df['label'].values, #test_df['label']. values))] For binary entropy loss function usage
history = model.fit(train_padded, y_train, epochs=10, batch_size=32)


# Plotting the accuracy-loss function graph curve for better insight.
def plot_loss_acc(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


plot_loss_acc(history)


model.save('res_ana.h5')