import  pandas as pd
import joblib
from flask import Flask
from flask import request
#import tokenizer from keras
from tensorflow.keras.preprocessing.text import Tokenizer
#import pad_sequence from keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def preprocess_predict(input):
  input = "eat your veggies: 9 deliciously different recipes"
  input  = re.sub('[^a-z ]+', '', input)
  input  = [i for i in input.split() if i not in stop_words]
  model = joblib.load("./model.pkl")
  tokeniser = joblib.load("./tokenizer.pkl")
  input = tokeniser.texts_to_sequences([input])
  padded_input = pad_sequences(input,maxlen = 120,truncating= 'post',padding='post')
  output  = model.predict(padded_input)
  if output> .5:
    sarcasm = True
  else:
    sarcasm = False
    return sarcasm


app = Flask(__name__)



@app.route('/predict', methods=['GET', 'POST'])
def add_message():
    #this will extract json
    input = request.get_json(force=True)
    
    prediction = preprocess_predict(input)
    
    return str(prediction)


app.run()