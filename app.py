from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from sklearn.model_selection import train_test_split
from model import the_model
import pickle
import numpy as np


app = Flask(__name__)
api = Api(app)

tmodel= the_model()

clf_path = 'SentimentClassifier.pkl'
with open(clf_path, 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    tmodel.clf = u.load()

vec_path = 'TFIDFVectorizer.pkl'
with open(vec_path, 'rb') as f:
    v= pickle._Unpickler(f)
    v.encoding = 'latin1'
    tmodel.vectorizer = v.load()


parser = reqparse.RequestParser()
parser.add_argument('query')



class PredictSentiment(Resource):
 
    def post(self):
        
        args = parser.parse_args()
        user_query = args['query']
        uq_vectorized = tmodel.vectorizer_transform(np.array([user_query]))
        prediction = tmodel.predict(uq_vectorized)
        pred_proba = tmodel.predict_proba(uq_vectorized)

        # Output either 'Negative' or 'Positive' 
        if prediction == 0:
            pred_text = 'Negative'
        else:
            pred_text = 'Positive'

        confidence = round(pred_proba[0], 3)

        # create JSON object
        output = {'prediction': pred_text, 'confidence': confidence}
        return output

    def get(self):
        return self.post()

        

api.add_resource(PredictSentiment, '/')


if __name__ == '__main__':
    app.run(debug=True)
