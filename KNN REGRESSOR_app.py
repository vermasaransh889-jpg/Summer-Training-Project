import streamlit as st
import pickle
import numpy as np

# Load Model
model = pickle.load(open("knn_regressor.pkl", "rb"))
scaler = pickle.load(open("scaler (1).pkl", "rb"))

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗")

st.title("🚗 Car Price Prediction using KNN Regressor")

# Inputs
name = st.number_input("Car Name (Encoded Value)", min_value=0)

year = st.number_input("Year", min_value=1990, max_value=2035, value=2018)

km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])

seller_type = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])

transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

owner = st.selectbox("Owner", ["First Owner",
                               "Second Owner",
                               "Third Owner",
                               "Fourth & Above Owner",
                               "Test Drive Car"])

# Manual Encoding
fuel_dict = {
    "CNG":0,
    "Diesel":1,
    "Electric":2,
    "LPG":3,
    "Petrol":4
}

seller_dict = {
    "Dealer":0,
    "Individual":1,
    "Trustmark Dealer":2
}

transmission_dict = {
    "Automatic":0,
    "Manual":1
}

owner_dict = {
    "First Owner":0,
    "Second Owner":1,
    "Third Owner":2,
    "Fourth & Above Owner":3,
    "Test Drive Car":4
}

fuel = fuel_dict[fuel]
seller_type = seller_dict[seller_type]
transmission = transmission_dict[transmission]
owner = owner_dict[owner]

if st.button("Predict Selling Price"):

    data = np.array([[name,
                      year,
                      km_driven,
                      fuel,
                      seller_type,
                      transmission,
                      owner]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    st.success(f"Predicted Selling Price: ₹ {prediction[0]:,.2f}")