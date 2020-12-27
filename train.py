"""Training process for dialogue genre classifier"""

import os
from argparse import ArgumentParser

import pandas as pd
from joblib import dump

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from preprocessing import generate_preprocessed_train_X_y

DEFAULT_TRAIN_DATASET_PATH = "./data/train.csv"
DEFAULT_TRAIN_FEATURES_PATH = "./data/preprocessed/train_features.csv"
DEFAULT_TRAIN_LABELS_PATH = "./data/preprocessed/train_labels.csv"
DEFAULT_MODEL_PATH = "model/model.pkl"
DEFAULT_VECTORIZER_PATH = "model/vectorizer.pkl"
DEFAULT_MLB_PATH = "model/multi_label_binarizer.pkl"


def train_model(model, train_X: pd.DataFrame, train_y: pd.DataFrame):
    """Function for model training.
       Returns fitted model and fitted word vectorizer."""
    word_vectorizer = TfidfVectorizer(min_df=3, stop_words='english')
    word_vectorizer.fit(train_X.dialogue)
    vectorized_train_X = word_vectorizer.transform(train_X.dialogue)
    model.fit(vectorized_train_X, train_y)

    return model, word_vectorizer


def save_model(model, vectorizer, mlb):
    """Function to save fitted model, TfidfVectorizer and
       MultiLabelBinarizer to files."""
    with open(DEFAULT_MODEL_PATH, "wb") as fin:
        dump(model, fin)
    with open(DEFAULT_VECTORIZER_PATH, "wb") as fin:
        dump(vectorizer, fin)
    with open(DEFAULT_MLB_PATH, "wb") as fin:
        dump(mlb, fin)


def callback_train(arguments):
    """Function to perform preprocessing and
       train process for given dataset."""
    train_df = pd.read_csv(arguments.dataset)
    genres_mlb = generate_preprocessed_train_X_y(
        train_df, DEFAULT_TRAIN_FEATURES_PATH, DEFAULT_TRAIN_LABELS_PATH)
    model = LogisticRegression(n_jobs=-1)
    clf = OneVsRestClassifier(model, n_jobs=-1)
    train_X = pd.read_csv(DEFAULT_TRAIN_FEATURES_PATH)
    train_y = pd.read_csv(DEFAULT_TRAIN_LABELS_PATH)
    clf, vectorizer = train_model(clf, train_X, train_y)
    if not os.path.exists("model"):
        os.makedirs("model")
    save_model(clf, vectorizer, genres_mlb)


def setup_parser(parser):
    """Function to setup CLI-arguments parser"""
    parser.set_defaults(callback=callback_train)
    parser.add_argument(
        "--dataset",
        default=DEFAULT_TRAIN_DATASET_PATH,
        help=f"path to csv file with train dataset, default is {DEFAULT_TRAIN_DATASET_PATH}"
    )


def main():
    """Main function."""
    parser = ArgumentParser(
        prog="Model train for movie genre classification",
        description="train ML model for movie dialogues genre classification",
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    arguments.callback(arguments)


if __name__ == "__main__":
    main()
