from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from prediction import *

app = Flask(__name__)
CORS(app)

# Load pre-trained deep learning model
model = load_model('FirstTry.h5')


# Define preprocessing function
def preprocess_text(text):
    # Apply any preprocessing steps here, such as removing punctuation, stopwords, etc.
    # Return the preprocessed text as a string
    pdf_file = convertPDFtoText(text)
    return pdf_file


# Define function to predict class from preprocessed text
def predict_class(text):
    # Convert text to vector using CountVectorizer or any other vectorizer that you prefer
    # vector = count_vectorizer.transform([text])

    tokened = tok(text)

    # Make prediction using pre-trained model
    prediction = model.predict(tokened)
    # Convert prediction to class label
    class_label = np.argmax(prediction)
    # Return class label as a string
    return str(class_label)


# Define API route for file upload
@app.route('/predict', methods=['POST'])
def predict():
    # Get uploaded file from request
    file = request.files['file']
    # Preprocess text data
    data = preprocess_text(file)
    # Make predictions for each text data point
    preds = predict_class(data)
    # Convert predicted class labels to list
    # predicted_classes = data['predicted_class'].tolist()
    # Return predicted classes as JSON response
    return jsonify({'predicted_classes': preds})


if __name__ == '__main__':
    app.run(debug=True)
