"""Describes Classifier class to build model for prediction"""

from joblib import load

MAGIC_THRESHOLD = 0.2559


class Classifier(object):
    """Classifier object for movie genre prediction on given dialogue:
       vectorizer - TfidfVectorizer
       model - OneVsRestClassifier over LogisticRegression
       mlb - MultiLabelBinarizer for encoding different genres"""

    def __init__(self, model_path, vectorizer_path, mlb_path):
        self.vectorizer = load(vectorizer_path)
        self.model = load(model_path)
        self.mlb = load(mlb_path)

    def predict(self, text):
        """Function to predict genres on given dialogue"""
        vectorized_text = self.vectorizer.transform([text])
        prediction = self.model.predict_proba(vectorized_text)
        prediction[prediction < MAGIC_THRESHOLD] = 0
        prediction[prediction >= MAGIC_THRESHOLD] = 1
        genres = self.mlb.inverse_transform(prediction)
        genres = [' '.join(x) for x in genres]

        return genres
