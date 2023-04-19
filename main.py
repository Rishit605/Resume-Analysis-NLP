from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from keras.models import load_model
import pdfplumber
import re

app = Flask(__name__)

# set the maximum file size to 16 megabytes
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# load the pre-trained Keras model
model = load_model('model.h5')

# define a list of class names
class_names = ['class1', 'class2', 'class3', ..., 'class25']

# define a function to preprocess the resume
def preprocess_resume(resume):
    # convert to lowercase
    resume = resume.lower()
    # remove non-word characters and extra spaces
    resume = re.sub(r'\W', ' ', resume)
    resume = re.sub(r'\s+', ' ', resume)
    return resume

# define a function to predict the class of the resume
def predict_class(resume):
    # preprocess the resume
    preprocessed_resume = preprocess_resume(resume)
    # tokenize the preprocessed resume
    tokenized_resume = tokenizer.texts_to_sequences([preprocessed_resume])
    # pad the tokenized resume to a fixed length
    padded_resume = pad_sequences(tokenized_resume, maxlen=max_length, padding='post', truncating='post')
    # make the prediction
    predictions = model.predict(padded_resume)
    # return the predicted class
    return class_names[np.argmax(predictions)]

# define the route to accept the file upload
@app.route('/classify', methods=['POST'])
def classify_resume():
    # check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    # get the file object from the request
    file = request.files['file']
    # check if the file is a PDF file
    if file.filename.split('.')[-1].lower() != 'pdf':
        return jsonify({'error': 'File must be a PDF file'}), 400
    # read the PDF file and extract the text
    with pdfplumber.load(file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    # predict the class of the resume
    predicted_class = predict_class(text)
    # return the predicted class
    return jsonify({'class': predicted_class}), 200

if __name__ == '__main__':
    app.run(debug=True)

    
    # Classify the resume using the pre-trained NLP model
    doc = nlp(text)
    labels = ['class1', 'class2', 'class3', ..., 'class25']
    scores = [doc.cats[label] for label in labels]
    predicted_class = labels[scores.index(max(scores))]
    
    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
