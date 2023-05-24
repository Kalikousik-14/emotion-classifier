from flask import Flask
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
import pickle

model = keras.models.load_model('model.h5')
loaded_vec = TfidfVectorizer(vocabulary=pickle.load(open("feature.pkl", "rb")))
grouped_values = ['admiration + amusement + exclamation + surprise', 'anger + annoyance','caring + love' ,'relief + realization + optimism' ,'curiosity + desire', 
                  'disappointment', 'disapproval + disgust + embarassment', 'fear + nervousness', 'approval + gratitude', 'grief + remorse + sadness', 
                  'confusion', 'joy', 'pride', 'neutral']
app = Flask(__name__)

def get_tfidf(sentences):
    transformer = TfidfTransformer()
    test_features = transformer.fit_transform(loaded_vec.fit_transform(np.array(sentences)))
    test_features = test_features.toarray()
    return np.reshape(test_features, (test_features.shape[0], 1, test_features.shape[1]))

@app.route('/getEmotion')
def outvalue():
    sentence = ['Yes I heard abt the f bombs! That has to be why. Thanks for your reply:) until then hubby and I will anxiously wait 😝']
    test_X = get_tfidf(sentence)
    y_prediction = model.predict(test_X)
    y_prediction = np.argmax (y_prediction, axis = 1)
    return grouped_values[y_prediction[0]]
 
if __name__ == '__main__':
    app.run()