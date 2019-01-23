from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
class the_model(object):
    
    def __init__(self):
        self.clf = MultinomialNB()
        self.vectorizer = TfidfVectorizer()

    def vectorizer_transform(self, X):
        X_transformed = self.vectorizer.transform(X)
        return X_transformed

    def predict_proba(self, X):
        y_proba = self.clf.predict_proba(X)
        return y_proba[:, 1]

    def predict(self, X):
        y_pred = self.clf.predict(X)
        return y_pred