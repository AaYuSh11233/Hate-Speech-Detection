from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
import joblib

def build_pipeline():
    return Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(random_state=42))
    ])

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
