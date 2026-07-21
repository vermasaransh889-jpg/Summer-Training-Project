import streamlit as st
import pickle
import pandas as pd

# Load trained model
model = pickle.load(open("titanic_rf_model.pkl", "rb"))

st.title("🚢 Titanic Survival Prediction")
st.write("Decision Tree Classifier")

# User Inputs
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["Male", "Female"])
Age = st.number_input("Age", min_value=0, max_value=100, value=25)
SibSp = st.number_input("Siblings/Spouses", min_value=0, max_value=10, value=0)
Parch = st.number_input("Parents/Children", min_value=0, max_value=10, value=0)
Fare = st.number_input("Fare", min_value=0.0, value=50.0)
Embarked = st.selectbox("Embarked", ["Q", "S", "C"])

# Encoding
Sex = 0 if Sex == "Male" else 1

Embarked_Q = 1 if Embarked == "Q" else 0
Embarked_S = 1 if Embarked == "S" else 0

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame(
        [[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_Q, Embarked_S]],
        columns=[
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked_Q",
            "Embarked_S",
        ],
    )

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")