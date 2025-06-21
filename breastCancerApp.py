import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# SHAP fallback
SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    pass

# Page settings
st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")

# Title
st.title("Breast Cancer Predictor")
st.markdown("Predict if a tumor is **Benign** or **Malignant** using different ML models.")

# Load model options
model_options = {
    "Random Forest": "random_forest_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
    "SVM": "svm_model.pkl",
    "KNN": "knn_model.pkl",
    "XGBoost": "xgboost_model.pkl"
}

model_choice = st.sidebar.selectbox("Choose Classifier", list(model_options.keys()))

# Load model
@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(model_options[model_choice])

# Load data for SHAP and CM
@st.cache_data
def load_data():
    df = pd.read_csv("wisconsin.csv")
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
    df["diagnosis"] = df["diagnosis"].map({"M": 0, "B": 1})  # 0 = Malignant, 1 = Benign
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    return X, y

X_data, y_data = load_data()

# Input sliders
st.sidebar.subheader("Input Features")
input_dict = {
    "mean radius": st.sidebar.slider("Mean Radius", 5.0, 30.0, 14.0),
    "mean texture": st.sidebar.slider("Mean Texture", 5.0, 40.0, 20.0),
    "mean perimeter": st.sidebar.slider("Mean Perimeter", 30.0, 200.0, 90.0),
    "mean area": st.sidebar.slider("Mean Area", 100.0, 2500.0, 500.0),
    "mean smoothness": st.sidebar.slider("Mean Smoothness", 0.05, 0.2, 0.1),
    "mean concavity": st.sidebar.slider("Mean Concavity", 0.0, 0.5, 0.1),
    "mean concave points": st.sidebar.slider("Mean Concave Points", 0.0, 0.2, 0.05),
    "worst radius": st.sidebar.slider("Worst Radius", 7.0, 40.0, 16.0),
    "worst texture": st.sidebar.slider("Worst Texture", 10.0, 50.0, 25.0),
    "worst perimeter": st.sidebar.slider("Worst Perimeter", 50.0, 250.0, 100.0),
}

input_df = pd.DataFrame([input_dict])

# Predict
prediction = model.predict(input_df)[0]
proba = model.predict_proba(input_df)[0]

st.subheader("🩺 Prediction Result")
st.success(f"**{'Benign' if prediction == 1 else 'Malignant'}** with {max(proba)*100:.2f}% confidence.")

# Confusion Matrix
if st.checkbox("Show Confusion Matrix"):
    y_pred = model.predict(X_data)
    cm = confusion_matrix(y_data, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

# SHAP with fallback
if SHAP_AVAILABLE and st.checkbox("Show SHAP Feature Importance"):
    try:
        explainer = shap.Explainer(model, X_data)
        shap_values = explainer(X_data)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.summary_plot(shap_values, X_data, show=False)
        st.pyplot(bbox_inches="tight")
    except Exception as e:
        st.warning("SHAP could not be generated for this model.")
elif not SHAP_AVAILABLE and st.checkbox("Show SHAP Feature Importance"):
    st.warning("SHAP is not available in this environment. It has been skipped to avoid deployment failure.")

st.markdown("---")
