from __future__ import annotations

"""Module consists of custon transformers for preprocessing text dialogues"""


import re
import ast


import pandas as pd


from sklearn.base import BaseEstimator, TransformerMixin


train_data = pd.read_csv("../data/train.csv")
test_data = pd.read_csv("../data/test.csv")


class CleanTextPreprocessor(BaseEstimator, TransformerMixin):
    """Transformer object for text preprocessing."""

    def __init__(self):
        self._html_tag = '<.*?>'
        self._whitespaces = re.compile(r"\s+")

    def fit(self, X: pd.DataFrame, y=None) -> CleanTextPreprocessor:
        """Returns instance of CleanTextPreprocessor.
           y remains for sklearn pipelines compatibility."""
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Removes unnecesary symbols from text:
           html tags, whitespaces, new line symbols, punctuation signs.
           y remains for sklearn pipelines compatibility."""
        text = X['dialogue'].copy().values
        for idx, _ in enumerate(text):
            text[idx] = re.sub(self._html_tag, ' ', text[idx])
            text[idx] = re.sub(r'[?|!|\'|"|#]', '', text[idx])
            text[idx] = re.sub(r'[,|.|;|:|(|)|{|}|\|/|<|>]|-', ' ', text[idx])
            text[idx] = text[idx].replace("\n", " ")
            text[idx] = re.sub('[^a-z A-Z]+', ' ', text[idx])
            text[idx] = text[idx].lower()
            text[idx] = self._whitespaces.sub(" ", text[idx]).strip()

        X.drop('dialogue', axis=1, inplace=True)
        text = pd.DataFrame(text, columns=['dialogue'])
        X = pd.concat([X, text], axis=1)

        return X


class GenresListPreprocessor(BaseEstimator, TransformerMixin):
    """Object for extracting genres list from dataset."""

    def __init__(self):
        self._genres_list = []

    def fit(self, X: pd.DataFrame, y=None) -> GenresListPreprocessor:
        """Extracts genres list from dataset.
           y remains for sklearn pipelines compatibility."""
        X['genres'] = X['genres'].apply(ast.literal_eval)
        for genre in X['genres']:
            self._genres_list.extend(genre)
        self._genres_list = list(set(self._genres_list))

        return self

    def transform(self, X: pd.DataFrame, y=None) -> list:
        """Returns list of genres.
           y remains for sklearn pipelines compatibility."""
        return X
