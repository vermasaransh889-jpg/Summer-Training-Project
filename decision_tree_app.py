import streamlit as st
import pickle
import pandas as pd
model=pickle.load(open("decision_tree_model.pkl", "rb"))
st.title("Decision Tree Salary Prediction")
exp=st.number_input("Enter Years of Experience", min_value=0.0)
if st.button("Predict"):
    salary=model.predict(pd.DataFrame([[exp]], columns=["YearsExperience"]))
    st.success(f"Predicted Salary:Rs{salary[0]:,.2f}")