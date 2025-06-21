import joblib
import numpy as np

def textModelLoad():
    model = joblib.load("random_forest_model.pkl")
    assert model is not None

def testModelPred():
    model = joblib.load("random_forest_model.pkl")
    sample = np.array([[14.0, 20.0, 90.0, 500.0, 0.1, 0.1, 0.05, 16.0, 25.0, 100.0]])
    prediction = model.predict(sample)
    assert prediction[0] in [0, 1]