"""Make genres prediction on given dialogue from movie"""
from flask import Flask
from classifier import Classifier
from train import DEFAULT_MODEL_PATH, DEFAULT_VECTORIZER_PATH, DEFAULT_MLB_PATH

app = Flask(__name__)

@app.route('/')
def index():
    """Main page of web app"""
    clf = Classifier(DEFAULT_MODEL_PATH,
                     DEFAULT_VECTORIZER_PATH, DEFAULT_MLB_PATH)
    dialogue = input()
    prediction = clf.predict(dialogue)
    print(" ".join(prediction))
