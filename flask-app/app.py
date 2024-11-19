from flask import Flask , render_template, request
import mlflow
import dagshub
from preprocessing_utility import *
import pickle

app = Flask(__name__)


dagshub.init(repo_owner='saurav-sabu', repo_name='mlops-mini-project', mlflow=True)

model_name = "my_model"
model_version = 3

model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)

vectorizer = pickle.load(open("models/vectorizer.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html",result=None)

@app.route("/predict",methods=["POST"])
def predict():
    text =  request.form["text"]

    # clean
    text = normalize_text(text)

    # bow
    features = vectorizer.transform([text])

    # prediction
    result = model.predict(features)

    return render_template("index.html",result = result[0])
    

if __name__ == "__main__":
    app.run(debug=True)