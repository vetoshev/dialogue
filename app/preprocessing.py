from __future__ import annotations

"""Module consists of methods for preprocessing train and test datasets"""


import os

import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from preprocessing_utility import CleanTextPreprocessor, GenresListPreprocessor


def generate_preprocessed_train_X_y(train_df: pd.DataFrame, filepath_features: str,
                                    filepath_labels: str) -> MultiLabelBinarizer:
    """Function for dataset preprocessing.
       Returns MultiLabelBinarizer object for future use.
       filepath_features - path to file, where preprocessed train features dataframe
       will be stored.
       filepath_labels - path to file, where preprocessed train labels dataframe
       will be stored."""
    text_cleaner = CleanTextPreprocessor()
    genres_creator = GenresListPreprocessor()
    train_df = text_cleaner.fit_transform(train_df)
    train_df = genres_creator.fit_transform(train_df)
    genres_mlb = MultiLabelBinarizer(classes=genres_creator._genres_list)
    genres_train_binary_df = pd.DataFrame(genres_mlb.fit_transform(
        train_df['genres']), columns=genres_creator._genres_list)
    train_df = pd.concat([train_df, genres_train_binary_df], axis=1)
    train_df.drop(['movie', 'genres'], axis=1, inplace=True)
    train_df_X = train_df.iloc[:, 1]
    train_df_y = train_df.iloc[:, 2:]
    if not os.path.exists("data/preprocessed"):
        os.makedirs("data/preprocessed")
    train_df_X.to_csv(filepath_features, index=False)
    train_df_y.to_csv(filepath_labels, index=False)

    return genres_mlb


def generate_preprocessed_test_X(test_df: pd.DataFrame, filepath_features: str):
    """Function for preprocessing test dataframe.
       filepath_features - path to file, where preprocessed test features dataframe
       will be stored."""
    text_cleaner = CleanTextPreprocessor()
    test_df_X = text_cleaner.fit_transform(test_df)
    test_df_X.to_csv(filepath_features, index=False)
