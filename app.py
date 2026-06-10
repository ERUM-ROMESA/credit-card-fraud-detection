from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open("../models/random_forest.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "Fraud Detection API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)

    return jsonify({
        "prediction": int(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)