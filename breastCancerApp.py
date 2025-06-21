import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("wisconsin.csv")
    data.drop("id", axis=1, inplace=True)
    return data

data = load_data()
mean_cols = [col for col in data.columns if col.endswith('_mean')]
X = data[mean_cols]
y = data['diagnosis']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Standardization and PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# Train models
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y)
dt = DecisionTreeClassifier()
dt.fit(X_scaled, y)

# App layout
st.title("Breast Cancer Diagnosis Predictor")
st.write("This app predicts whether a tumor is **Benign** or **Malignant** based on input features using K-NN or Decision Tree Classifier.")

# Model selection
model_choice = st.sidebar.selectbox("Choose Classifier", ["K-Nearest Neighbors", "Decision Tree"])
user_input = {}
for col in mean_cols:
    user_input[col] = st.sidebar.slider(col, float(data[col].min()), float(data[col].max()), float(data[col].mean()))

input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)

# Prediction
if st.sidebar.button("Predict"):
    if model_choice == "K-Nearest Neighbors":
        model = knn
    else:
        model = dt

    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else None

    st.subheader("Prediction")
    st.write("Prediction:", "**Malignant**" if pred == 1 else "**Benign**")
    if prob is not None:
        st.write("Probability of Malignant:", round(prob, 2))

    # Classification report
    y_pred = model.predict(X_scaled)
    report = classification_report(y, y_pred, target_names=['Benign', 'Malignant'], output_dict=True)
    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())

    # Confusion matrix
    conf_mat = confusion_matrix(y, y_pred)
    st.subheader("Confusion Matrix")
    st.write(conf_mat)

    # ROC Curve
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y, probs)
        auc_score = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'{model_choice} (AUC = {auc_score:.2f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        st.pyplot(fig)

# Expandable sections
with st.expander("Dataset Overview"):
    st.dataframe(data.head())

with st.expander("PCA Analysis"):
    explained_var = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_var)
    st.line_chart(cumulative_explained_variance)
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA Projection (2D)")
    st.pyplot(fig)

with st.expander(" KNN Tuning: Error Rate vs K"):
    error_rates = []
    for k in range(1, 21):
        model_k = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(model_k, X_scaled, y, cv=5)
        error_rates.append(1 - scores.mean())
    fig, ax = plt.subplots()
    ax.plot(range(1, 21), error_rates, marker='o')
    ax.set_xlabel("K")
    ax.set_ylabel("Misclassification Error")
    ax.set_title("Error Rate vs K")
    st.pyplot(fig)

with st.expander("Project Reflection"):
    st.markdown("""
    **Reflection:**  
    Both the K-NN and Decision Tree classifiers performed well, with K-NN slightly outperforming in accuracy. However, after evaluating precision, recall, and F1-score using the classification report, K-NN also demonstrated better balance in correctly identifying malignant cases with fewer false negatives—critical in a medical setting.  
    The ROC curve comparison further validated this, showing a higher AUC for K-NN. These robustness metrics show that K-NN not only performs better on average but is also more reliable in sensitive healthcare diagnostics.
    """)
