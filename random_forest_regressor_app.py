import streamlit as st
import pandas as pd
import pickle

# Set page title and layout
st.set_page_config(page_title="Insurance Cost Predictor", page_icon="💳", layout="centered")

st.title("💳 Medical Insurance Cost Prediction")
st.write("Fill in the fields below to estimate annual medical charges using your Random Forest Regressor.")

st.divider()

# 1. Load trained model
@st.cache_resource
def load_model():
    with open('random_forest_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file `random_forest_model.pkl` not found in current folder!")
    st.stop()

# 2. Input Fields based on dataset columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=25, step=1)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)

with col2:
    sex = st.selectbox("Sex", ["female", "male"])
    smoker = st.selectbox("Smoker?", ["no", "yes"])
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

st.divider()

# 3. Prediction Button & Handling One-Hot Encoding
if st.button("Predict Charges", type="primary", use_container_width=True):
    
    # Map raw user inputs to match pd.get_dummies(df, drop_first=True)
    input_data = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex_male': 1 if sex == 'male' else 0,
        'smoker_yes': 1 if smoker == 'yes' else 0,
        'region_northwest': 1 if region == 'northwest' else 0,
        'region_southeast': 1 if region == 'southeast' else 0,
        'region_southwest': 1 if region == 'southwest' else 0,
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Predict
    predicted_charge = model.predict(input_df)[0]

    # Display Result
    st.success(f"### Estimated Medical Cost: **${predicted_charge:,.2f}**")