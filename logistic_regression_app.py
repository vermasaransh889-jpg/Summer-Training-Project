import streamlit as st
import pickle
from sklearn.datasets import load_iris

# Load model
model = pickle.load(open("Logistic_Regression_Iris.pkl", "rb"))

# Load Iris target names
iris = load_iris()

st.title("🌸 Iris Flower Prediction")
st.write("Enter the flower measurements below:")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", value=3.5)
petal_length = st.number_input("Petal Length (cm)", value=1.4)
petal_width = st.number_input("Petal Width (cm)", value=0.2)

# Prediction
if st.button("Predict"):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    flower = iris.target_names[prediction[0]]

    st.success(f"Predicted Flower: {flower}")