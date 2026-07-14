import streamlit as st
import pickle
model=pickle.load(open("Untitled0.pkl","rb"))
st.title("house price prediction")
sqft=st.number_input('Enter Living Area("sqft")')
if st.button("predict"):
    price= model.predict([[sqft]])
    st.write("Predicted Price:", price[0])