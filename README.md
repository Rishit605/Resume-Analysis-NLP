# Resume-Analyser: NLP-based Resume Classification System

Overview:
The NLP-based Resume Classification System is a machine learning project that uses natural language processing techniques to classify resumes into 25 different categories based on the job position and qualifications. The system is built using Python, TensorFlow-Keras, and NLTK libraries. The system is hosted on a local server that accepts input resumes in PDF form using a Flask API.

## Dataset
The dataset consists of 15k labelled resumes (labelled according to the primary category/class that a particular resume belongs to) in a csv format. We will be using this csv formatted resume dataset to train our model for classification. Our model should then be able to work on any unseen resume and some other data that was picked form diffrent sources likely Kaggle etc.

## Features:

* Classifies resumes into 43 different categories based on job position and qualifications.
* Accepts input resumes in PDF form using a Flask API.
* Uses TensorFlow-Keras and NLTK libraries for natural language processing and machine learning.
* Hosted on a local server, making it easily accessible to the user.

## Technical Details:
The NLP-based Resume Classification System is built using Python and several popular libraries, including TensorFlow-Keras and NLTK. The system utilizes deep learning algorithms to analyze resumes and classify them into relevant job categories. The system also preprocesses resumes using NLTK to remove stop words, perform stemming, and tokenization.

To use the system, users simply upload a resume in PDF format to the Flask API hosted on the local server. The system then uses a pre-trained TensorFlow-Keras model to classify the resume into one of 25 categories based on the job position and qualifications. The user can then view the classification result on the website.

# Conclusion:
The NLP-based Resume Classification System is an innovative solution for recruiters and HR departments looking to streamline the resume classification process. By using natural language processing and deep learning algorithms, the system can quickly and accurately classify resumes into relevant job categories, saving time and effort for recruiters.
