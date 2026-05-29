import os
import pickle

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# =========================================
# CUSTOM CSS
# =========================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

h1, h2, h3 {
    color: #00F5FF;
}

.stButton>button {
    background-color: #00F5FF;
    color: black;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

.stSlider > div > div > div > div {
    background-color: #00F5FF;
}

.prediction-box {
    padding: 25px;
    border-radius: 12px;
    background-color: #111827;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    color: white;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# TITLE
# =========================================

st.title("🚢 Titanic Survival Prediction using AdaBoost")

st.markdown("---")

# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv(
    "data/Titanic-Dataset.csv"
)

# =========================================
# DATA PREVIEW
# =========================================

st.subheader("📊 Dataset Preview")

st.dataframe(
    df.head()
)

# =========================================
# DATASET STATISTICS
# =========================================

st.subheader("📋 Dataset Statistics")

st.dataframe(
    df.describe()
)

# =========================================
# HANDLE MISSING VALUES
# =========================================

df["Age"] = df["Age"].fillna(
    df["Age"].mean()
)

df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)

df["Fare"] = df["Fare"].fillna(
    df["Fare"].mean()
)

# =========================================
# DROP UNUSED COLUMNS
# =========================================

df.drop(
    ["PassengerId", "Name", "Ticket", "Cabin"],
    axis=1,
    inplace=True
)

# =========================================
# LABEL ENCODING
# =========================================

le_sex = LabelEncoder()
le_embarked = LabelEncoder()

df["Sex"] = le_sex.fit_transform(
    df["Sex"]
)

df["Embarked"] = le_embarked.fit_transform(
    df["Embarked"]
)

# =========================================
# SIDEBAR HYPERPARAMETERS
# =========================================

st.sidebar.header("⚙️ Hyperparameters")

n_estimators = st.sidebar.slider(
    "Number of Estimators",
    10,
    500,
    100
)

learning_rate = st.sidebar.slider(
    "Learning Rate",
    0.01,
    2.0,
    1.0
)

max_depth = st.sidebar.slider(
    "Base Estimator Max Depth",
    1,
    20,
    3
)

test_size = st.sidebar.slider(
    "Test Size",
    0.1,
    0.4,
    0.2
)

random_state = st.sidebar.slider(
    "Random State",
    1,
    100,
    42
)

# =========================================
# FEATURES & TARGET
# =========================================

x = df.drop(
    "Survived",
    axis=1
)

y = df["Survived"]

# =========================================
# TRAIN TEST SPLIT
# =========================================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=test_size,
    random_state=random_state
)

# =========================================
# BASE ESTIMATOR
# =========================================

base_estimator = DecisionTreeClassifier(
    max_depth=max_depth
)

# =========================================
# MODEL
# =========================================

model = AdaBoostClassifier(

    estimator=base_estimator,

    n_estimators=n_estimators,

    learning_rate=learning_rate,

    random_state=random_state
)

# =========================================
# TRAIN MODEL
# =========================================

model.fit(
    x_train,
    y_train
)

# =========================================
# SAVE MODEL
# =========================================

os.makedirs(
    "models",
    exist_ok=True
)

with open(
    "models/adaboost_classifier.pkl",
    "wb"
) as f:

    pickle.dump(model, f)

# =========================================
# LOAD MODEL
# =========================================

with open(
    "models/adaboost_classifier.pkl",
    "rb"
) as f:

    loaded_model = pickle.load(f)

# =========================================
# PREDICTIONS
# =========================================

y_pred = loaded_model.predict(
    x_test
)

# =========================================
# MODEL PERFORMANCE
# =========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

st.subheader("📊 Model Performance")

c1, c2 = st.columns(2)

c1.metric(
    "Accuracy",
    f"{accuracy:.4f}"
)

c2.metric(
    "Features",
    x.shape[1]
)

# =========================================
# CORRELATION HEATMAP
# =========================================

st.subheader("🔥 Correlation Heatmap")

numeric_df = df.select_dtypes(
    include=np.number
)

fig_corr, ax_corr = plt.subplots(
    figsize=(10,8)
)

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    ax=ax_corr
)

st.pyplot(fig_corr)

# =========================================
# FEATURE IMPORTANCE
# =========================================

st.subheader("🌟 Feature Importance")

importance = pd.DataFrame({

    "Feature": x.columns,

    "Importance": loaded_model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

fig_imp, ax_imp = plt.subplots(
    figsize=(8,6)
)

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature",
    ax=ax_imp
)

st.pyplot(fig_imp)

# =========================================
# CONFUSION MATRIX
# =========================================

st.subheader("🧩 Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred
)

fig_cm, ax_cm = plt.subplots(
    figsize=(5,4)
)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=ax_cm
)

ax_cm.set_xlabel("Predicted")

ax_cm.set_ylabel("Actual")

st.pyplot(fig_cm)

# =========================================
# CLASSIFICATION REPORT
# =========================================

st.subheader("📄 Classification Report")

report = classification_report(
    y_test,
    y_pred
)

st.text(report)


# =========================================
# PREDICTION SECTION
# =========================================

st.markdown("---")

st.subheader("📝 Predict Survival")

# =========================================
# USER INPUTS
# =========================================

pclass = st.slider(
    "Passenger Class",
    1,
    3,
    1
)

sex = st.selectbox(
    "Sex",
    le_sex.classes_
)

age = st.slider(
    "Age",
    1,
    80,
    25
)

sibsp = st.slider(
    "Siblings / Spouses",
    0,
    10,
    0
)

parch = st.slider(
    "Parents / Children",
    0,
    10,
    0
)

fare = st.slider(
    "Fare",
    0.0,
    600.0,
    50.0
)

embarked = st.selectbox(
    "Embarked",
    le_embarked.classes_
)

# =========================================
# ENCODE INPUTS
# =========================================

sex_encoded = le_sex.transform(
    [sex]
)[0]

embarked_encoded = le_embarked.transform(
    [embarked]
)[0]

# =========================================
# INPUT DATAFRAME
# =========================================

input_df = pd.DataFrame([{

    "Pclass": pclass,

    "Sex": sex_encoded,

    "Age": age,

    "SibSp": sibsp,

    "Parch": parch,

    "Fare": fare,

    "Embarked": embarked_encoded

}])

# =========================================
# PREDICT BUTTON
# =========================================

if st.button("Predict Survival"):

    prediction = loaded_model.predict(
        input_df
    )[0]

    probability = loaded_model.predict_proba(
        input_df
    )[0]

    st.markdown("---")

    if prediction == 1:

        st.markdown(f"""
        <div class="prediction-box">
            ✅ Passenger Will Survive
            <br><br>
            Survival Probability :
            {probability[1]*100:.2f}%
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="prediction-box">
            ❌ Passenger Will Not Survive
            <br><br>
            Survival Probability :
            {probability[0]*100:.2f}%
        </div>
        """, unsafe_allow_html=True)