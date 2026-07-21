import streamlit as st
import pickle
import numpy as np

# Load Model and Scaler
model = pickle.load(open("knn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

st.title("❤️ Heart Disease Prediction using KNN")

# Numerical Inputs
age = st.number_input("Age", min_value=1, max_value=120, value=30)

gender = st.selectbox("Gender", ["Male", "Female"])

weight = st.number_input("Weight (kg)", value=70.0)
height = st.number_input("Height (cm)", value=170.0)
bmi = st.number_input("BMI", value=24.0)

smoking = st.selectbox("Smoking", ["Never", "Former", "Current"])

alcohol = st.selectbox("Alcohol Intake", ["None", "Low", "Moderate", "High"])

physical = st.selectbox("Physical Activity", ["Sedentary", "Moderate", "Active"])

diet = st.selectbox("Diet", ["Healthy", "Average", "Unhealthy"])

stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])

hypertension = st.selectbox("Hypertension", [0,1])

diabetes = st.selectbox("Diabetes", [0,1])

hyperlipidemia = st.selectbox("Hyperlipidemia", [0,1])

family = st.selectbox("Family History", [0,1])

previous = st.selectbox("Previous Heart Disease", [0,1])

sys_bp = st.number_input("Systolic BP", value=120)

dia_bp = st.number_input("Diastolic BP", value=80)

heart_rate = st.number_input("Heart Rate", value=72)

blood_sugar = st.number_input("Blood Sugar", value=100)

cholesterol = st.number_input("Cholesterol", value=180)

# Encoding
gender = 1 if gender=="Male" else 0

smoking = {"Current":0,"Former":1,"Never":2}[smoking]

alcohol = {"None":0,"Low":1,"Moderate":2,"High":3}[alcohol]

physical = {"Sedentary":0,"Moderate":1,"Active":2}[physical]

diet = {"Healthy":0,"Average":1,"Unhealthy":2}[diet]

stress = {"Low":0,"Medium":1,"High":2}[stress]

if st.button("Predict"):

    data = np.array([[age,gender,weight,height,bmi,
                      smoking,alcohol,physical,diet,stress,
                      hypertension,diabetes,hyperlipidemia,
                      family,previous,sys_bp,dia_bp,
                      heart_rate,blood_sugar,cholesterol]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")