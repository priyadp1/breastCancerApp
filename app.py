from flask import Flask, request, jsonify
import joblib
import numpy as np
import os


app = Flask(__name__)
MODELPATHS = {
    "random_forest": "random_forest_model.pkl",
    "logistic_regression": "logistic_regression_model.pkl",
    "svm": "svm_model.pkl",
    "knn": "knn_model.pkl",
    "xgboost": "xgboost_model.pkl"
}

MODELS = {name: joblib.load(path) for name, path in MODELPATHS.items()}

@app.route("/")
def home():
    return "Backend is live"


@app.route("/predict" , methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array([list(data.values())].astype(float))
        results = {}
        for modelName , model in MODELS.items():
            prediction = int(model.predict(features)[0])
            confidence = round(float(np.max(model.predict_proba(features))) * 100, 2)
            results[modelName] = {
                "prediction": prediction,
                "confidence": confidence
            }
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": str(e)}) , 500
    


if __name__ == "__main__":
    app.run(debug=True)

