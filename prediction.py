# -*- coding: utf-8 -*-
"""Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CVaPMmUwI5CAKW-DJDYXnfLcl2dK63Ke

# **PLEASE INSTALL THESE LIBRARIES**
"""

# ! pip install -r requirements.txt
#
# ! pip install pdfminer.six==20181108
#
# ! pip install keras_preprocessing

from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import numpy as np
import nltk
from nltk.probability import FreqDist
from string import punctuation
import math
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

nltk.download('punkt')
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')


# Converting a PDF file to text
# def convertPDFtoText(path):
#     rsrcmgr = PDFResourceManager()
#     retstr = StringIO()
#     codec = 'utf-8'
#     laparams = LAParams()
#     device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
#     fp = open(path, 'rb')
#     interpreter = PDFPageInterpreter(rsrcmgr, device)
#     password = ""
#     maxpages = 0
#     caching = True
#     pagenos = set()
#     for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching,
#                                   check_extractable=True):
#         interpreter.process_page(page)
#     fp.close()
#     device.close()
#     string = retstr.getvalue()
#     retstr.close()
#     return string

def convertPDFtoText(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos = set()
    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching,
                                  check_extractable=True):
        interpreter.process_page(page)
    fp.close()
    device.close()
    string = retstr.getvalue()
    retstr.close()
    return string




test_resume = convertPDFtoText('/mohit resume.pdf')
print(test_resume)


def summarize(doc, words):
    score = {}
    fd = FreqDist(words)
    for i, t in enumerate(doc):
        score[i] = 0
        for j in nltk.word_tokenize(t):
            if j in fd:
                score[i] += fd[j]

    r = sorted(list(score.items()), key=lambda x: x[1], reverse=True)[:math.floor(0.60 * len(doc))]
    r.sort(key=lambda x: x[0])
    l = [doc[i[0]] for i in r]
    return "\n\n".join(l)


import string


def rem_punc(s):
    punc = string.punctuation
    return [i for i in s if i not in punc]


def rem_sw(s):
    sw = set(stopwords)
    return [i for i in s if i not in sw]


def preprocess(eval_res):
    try:
        eval_res = eval(eval_res).decode()
    except:
        pass
    eval_res = eval_res.encode("ASCII", "ignore").decode()
    length = len(eval_res)
    eval_res = " ".join(eval_res.split("\n"))
    token = rem_sw(
        nltk.word_tokenize(eval_res))  # Removing punctaution later since we need punctaution for sentence tokenization
    eval_res = " ".join(token).lower()
    return eval_res


resume = preprocess(test_resume)  # remove stop words etc
sent = nltk.sent_tokenize(test_resume)
puncu = punctuation
word_token = nltk.word_tokenize(test_resume)  # tokenize preprocessed text for scoring

print(summarize(sent, test_resume))

word_token


# Tokenizing the given text
def tok(text):
    token = Tokenizer(num_words=10000, oov_token="<OOV>")
    token.fit_on_texts(text)

    word_idx = token.word_index

    train_seq = token.texts_to_sequences(text)
    train_pad = pad_sequences(train_seq, maxlen=120, truncating="pre")

    return train_pad


# Hard-Coding the Classes
def classes():
    original_dict = {'Accountant': 0, 'Advocate': 1, 'Agricultural': 2, 'Apparel': 3, 'Architects': 4,
                     'Arts': 5, 'Automobile': 6, 'Aviation': 7, 'BPO': 8, 'Banking': 9,
                     'Building & Construction': 10, 'Business Development': 11, 'Consultant': 12,
                     'Designing': 13, 'Digital Media': 14, 'Education': 15, 'Engineering': 16,
                     'Finance': 17, 'Food & Beverages': 18, 'HR': 19, 'Health & Fitness': 20,
                     'Information Technology': 21, 'Managment': 22, 'Public Relations': 23, 'Sales': 24}
    reversed_dict = {v: k for k, v in original_dict.items()}
    return reversed_dict


classes()


# Prediction Function
def prediction():
    model = load_model('/FirstTry.h5')
    pred = model.predict(tok(test_resume))
    predicted = np.argmax(pred, axis=-1)[0]
    Cls = classes()[predicted]
    # return (f" Your Predicted Class is of the {predicted} index with the name '{Cls}'")
    return predicted, Cls


prediction()
