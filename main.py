from flask import Flask, request, jsonify

import tensorflow as tf
import numpy as np
from prediction import  convertPDFtoText, tok, classes

app = Flask(__name__)

# Load the pre-trained Keras model
model = tf.keras.models.load_model('res_ana.h5')

# Define the mapping of class indices to class names
class_names = classes()


@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file from the request
    file = request.files['file']

    # Convert the PDF file to text
    text = convertPDFtoText(file)

    # Preprocess the text
    # ...
    preprocessed_text = tok(text)
    # Use the pre-trained model to make a prediction
    # ...
    prediction = model.predict(preprocessed_text)

    # Convert the prediction to a class name
    class_index = np.argmax(prediction)
    class_name = class_names[class_index]

    # Return the prediction as JSON
    return jsonify({'class': class_name})


if __name__ == '__main__':
    app.run(debug=True)
