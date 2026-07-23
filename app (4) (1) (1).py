"""
Fitness Tracker - Calorie Burn Predictor + Daily Nutrition Planner
--------------------------------------------------------------------
A Streamlit web app that:
  1. Uses a trained RandomForestRegressor model to predict calories
     burnt during a workout, based on personal + exercise details.
  2. Calculates daily calorie needs (BMR/TDEE) and gives a simple
     diet/macro recommendation based on the user's goal.

Model features (in this exact order):
    id, Sex, Age, Height, Weight, Duration, Heart_Rate, Body_Temp, BMI, workout type

Author: Miku
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st

# =================================================
# Page configuration (must be the first st. command)
# =================================================
st.set_page_config(
    page_title="Fitness Tracker",
    page_icon="🔥",
    layout="centered",
    initial_sidebar_state="expanded",
)

# =================================================
# Custom styling
# =================================================
st.markdown("""
<style>
    /* ===== Black + Blue base theme ===== */
    .stApp {
        background: linear-gradient(180deg, #000000 0%, #050b18 40%, #0a1a2f 100%);
    }

    .main .block-container {padding-top: 2rem; padding-bottom: 3rem;}

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #00040a 0%, #061224 100%);
        border-right: 1px solid rgba(59, 130, 246, 0.25);
    }

    /* Make virtually all text bright white and readable */
    html, body, [class*="css"],
    .stApp, .stApp p, .stApp span, .stApp label, .stApp li,
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
    .stCaption, [data-testid="stCaptionContainer"],
    h1, h2, h3, h4, h5, h6,
    div[data-testid="stMetricLabel"], div[data-testid="stMetricValue"], div[data-testid="stMetricDelta"],
    .stRadio label, .stSelectbox label, .stSlider label, .stTabs [data-baseweb="tab"] {
        color: #FFFFFF !important;
    }

    h1 {
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(59, 130, 246, 0.08);
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(59, 130, 246, 0.25) !important;
        border-radius: 8px;
    }

    div[data-testid="stMetric"] {
        background: rgba(59, 130, 246, 0.10);
        border: 1px solid rgba(59, 130, 246, 0.30);
        border-radius: 12px;
        padding: 14px 12px;
    }

    div[data-testid="stMetricValue"] {
        font-weight: 700;
        color: #FFFFFF !important;
    }

    .diet-card {
        background: rgba(59, 130, 246, 0.10);
        border: 1px solid rgba(59, 130, 246, 0.30);
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
        color: #FFFFFF;
    }

    .diet-card h4 {
        margin-top: 0;
        margin-bottom: 8px;
        color: #FFFFFF !important;
    }

    .diet-card p, .diet-card li {
        color: #FFFFFF !important;
    }

    .macro-bar-bg {
        background: rgba(59, 130, 246, 0.15);
        border-radius: 8px;
        height: 10px;
        width: 100%;
        overflow: hidden;
        margin-top: 4px;
        margin-bottom: 2px;
    }

    .macro-bar-fill {
        height: 10px;
        border-radius: 8px;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        background: linear-gradient(90deg, #1d4ed8, #3b82f6);
        color: #FFFFFF !important;
        border: none;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #2563eb, #60a5fa);
        color: #FFFFFF !important;
    }

    /* Dataframe / table text */
    .stDataFrame, .stDataFrame div {
        color: #FFFFFF !important;
    }

    /* Divider color to match theme */
    hr {
        border-color: rgba(59, 130, 246, 0.3) !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        color: #FFFFFF !important;
        background: rgba(59, 130, 246, 0.08);
    }

    /* ===== Remove the white top header bar so it matches the app theme ===== */
    header[data-testid="stHeader"] {
        background: #000000 !important;
        background-image: none !important;
    }

    /* Remove the colored decoration line Streamlit renders at the very top */
    div[data-testid="stDecoration"] {
        background: #000000 !important;
        background-image: none !important;
        display: none !important;
    }

    /* Toolbar icons (top-right menu) stay visible on the dark background */
    header[data-testid="stHeader"] * {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# =================================================
# Load the trained model (cached so it loads only once)
# =================================================
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# =================================================
# Header
# =================================================
st.title("🔥 Fitness Tracker")
st.caption("Predict calories burnt during your workout, and see your full daily nutrition picture.")

# =================================================
# Sidebar - user inputs
# =================================================
st.sidebar.header("👤 Your Details")

sex = st.sidebar.radio("Sex", options=["Male", "Female"], horizontal=True)
age = st.sidebar.slider("Age (years)", 10, 80, 25)
height = st.sidebar.slider("Height (cm)", 120.0, 220.0, 170.0, step=0.5)
weight = st.sidebar.slider("Weight (kg)", 30.0, 150.0, 65.0, step=0.5)

st.sidebar.divider()
st.sidebar.header("🏋️ Workout Details")

workout_type = st.sidebar.selectbox("Workout Type", options=["HIIT", "Cardio"])
duration = st.sidebar.slider("Duration (minutes)", 1, 120, 20)
heart_rate = st.sidebar.slider("Heart Rate (bpm)", 60, 180, 100)
body_temp = st.sidebar.slider("Body Temperature (°C)", 36.0, 42.0, 39.0, step=0.1)

predict_btn = st.sidebar.button("Predict Calories Burnt", use_container_width=True, type="primary")

st.sidebar.divider()
st.sidebar.header("🎯 Nutrition Goal")

activity_level = st.sidebar.selectbox(
    "Daily Activity Level",
    options=[
        "Sedentary (little/no exercise)",
        "Light (1-3 days/week)",
        "Moderate (3-5 days/week)",
        "Active (6-7 days/week)",
        "Very Active (hard exercise/physical job)",
    ],
    index=2,
)
goal = st.sidebar.radio("Goal", options=["Lose Weight", "Maintain", "Gain Muscle"], horizontal=False)

# =================================================
# Auto-calculate BMI from height & weight
# =================================================
height_m = height / 100
bmi = weight / (height_m ** 2)

if bmi < 18.5:
    bmi_category = "Underweight"
elif bmi < 25:
    bmi_category = "Normal"
elif bmi < 30:
    bmi_category = "Overweight"
else:
    bmi_category = "Obese"

# =================================================
# BMR (Mifflin-St Jeor) + TDEE + goal-adjusted target
# =================================================
ACTIVITY_FACTORS = {
    "Sedentary (little/no exercise)": 1.2,
    "Light (1-3 days/week)": 1.375,
    "Moderate (3-5 days/week)": 1.55,
    "Active (6-7 days/week)": 1.725,
    "Very Active (hard exercise/physical job)": 1.9,
}

if sex == "Male":
    bmr = 10 * weight + 6.25 * height - 5 * age + 5
else:
    bmr = 10 * weight + 6.25 * height - 5 * age - 161

tdee = bmr * ACTIVITY_FACTORS[activity_level]

GOAL_ADJUSTMENT = {
    "Lose Weight": -500,
    "Maintain": 0,
    "Gain Muscle": 350,
}
daily_target = tdee + GOAL_ADJUSTMENT[goal]

# Macro split (grams) - simple, goal-based ratios of total target calories
MACRO_RATIOS = {
    "Lose Weight": {"protein": 0.35, "carbs": 0.35, "fat": 0.30},
    "Maintain":    {"protein": 0.30, "carbs": 0.40, "fat": 0.30},
    "Gain Muscle": {"protein": 0.30, "carbs": 0.45, "fat": 0.25},
}
ratios = MACRO_RATIOS[goal]
protein_g = (daily_target * ratios["protein"]) / 4
carbs_g = (daily_target * ratios["carbs"]) / 4
fat_g = (daily_target * ratios["fat"]) / 9

# =================================================
# Quick stats row
# =================================================
c1, c2, c3 = st.columns(3)
c1.metric("BMI", f"{bmi:.1f}", bmi_category)
c2.metric("Workout Duration", f"{duration} min")
c3.metric("Heart Rate", f"{heart_rate} bpm")

st.divider()

# =================================================
# Tabs: Calorie Predictor  |  Daily Nutrition Plan
# =================================================
tab_predict, tab_diet = st.tabs(["🔥 Calorie Burn Predictor", "🥗 Daily Nutrition Plan"])

# -------------------------------------------------
# TAB 1 — Calorie burn prediction (original feature)
# -------------------------------------------------
with tab_predict:
    # Encoding used while training the model:
    #   Sex: male -> 0, female -> 1
    #   workout type: HIIT -> 0, Cardio -> 1
    sex_encoded = 0 if sex == "Male" else 1
    workout_encoded = 0 if workout_type == "HIIT" else 1

    # 'id' was left in the training features by mistake in the original notebook,
    # it has no real predictive meaning, so we just pass a constant placeholder.
    input_df = pd.DataFrame([{
        "id": 0,
        "Sex": sex_encoded,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "BMI": bmi,
        "workout type": workout_encoded,
    }])

    if predict_btn:
        prediction = model.predict(input_df)[0]

        st.subheader("Result")
        pct_of_target = (prediction / daily_target) * 100 if daily_target else 0
        rc1, rc2 = st.columns(2)
        rc1.metric("Estimated Calories Burnt", f"{prediction:.0f} kcal")
        rc2.metric("% of Daily Calorie Target", f"{pct_of_target:.1f}%")

        with st.expander("Input Summary", expanded=False):
            summary_df = pd.DataFrame({
                "Parameter": ["Sex", "Age", "Height", "Weight", "BMI", "Workout Type",
                              "Duration", "Heart Rate", "Body Temp"],
                "Value": [sex, f"{age} yrs", f"{height} cm", f"{weight} kg",
                          f"{bmi:.1f} ({bmi_category})", workout_type,
                          f"{duration} min", f"{heart_rate} bpm", f"{body_temp} °C"]
            })
            st.dataframe(summary_df, hide_index=True, use_container_width=True)

        st.subheader("Calories Burnt vs. Duration")
        st.caption("How calories change as duration increases, keeping everything else the same.")

        # Show how calories would change across a range of durations,
        # keeping every other input the same as what the user entered.
        durations_range = np.arange(5, 121, 5)
        chart_rows = []
        for d in durations_range:
            row = input_df.copy()
            row["Duration"] = d
            chart_rows.append(row)
        chart_input = pd.concat(chart_rows, ignore_index=True)
        chart_preds = model.predict(chart_input)

        chart_df = pd.DataFrame({
            "Duration (min)": durations_range,
            "Predicted Calories": chart_preds
        }).set_index("Duration (min)")

        st.line_chart(chart_df)

    else:
        st.info("Fill in your details in the sidebar and click **Predict Calories Burnt** to see your result.")

# -------------------------------------------------
# TAB 2 — Daily calorie needs & diet recommendation
# -------------------------------------------------
with tab_diet:
    st.subheader("Your Daily Calorie Needs")

    d1, d2, d3 = st.columns(3)
    d1.metric("BMR", f"{bmr:.0f} kcal", help="Calories your body burns at complete rest.")
    d2.metric("Maintenance (TDEE)", f"{tdee:.0f} kcal", help="Calories burnt per day at your activity level.")
    d3.metric(f"Target ({goal})", f"{daily_target:.0f} kcal",
              delta=f"{GOAL_ADJUSTMENT[goal]:+d} kcal" if GOAL_ADJUSTMENT[goal] != 0 else "on maintenance")

    st.caption(
        "BMR is calculated using the Mifflin-St Jeor equation. TDEE scales BMR by your selected "
        "activity level, and the target adjusts TDEE for your goal (a ~500 kcal deficit/surplus "
        "is a commonly used, moderate rate of weight change)."
    )

    st.divider()
    st.subheader("Suggested Macronutrient Split")

    macro_colors = {"Protein": "#ff512f", "Carbs": "#f09819", "Fat": "#4facfe"}
    macro_data = [
        ("Protein", protein_g, ratios["protein"]),
        ("Carbs", carbs_g, ratios["carbs"]),
        ("Fat", fat_g, ratios["fat"]),
    ]
    for label, grams, pct in macro_data:
        st.markdown(f"**{label}** — {grams:.0f} g  ({pct*100:.0f}% of calories)")
        st.markdown(
            f"""<div class="macro-bar-bg">
                    <div class="macro-bar-fill" style="width:{pct*100:.0f}%; background:{macro_colors[label]};"></div>
                </div>""",
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Diet Recommendations")

    # Recommendations combine BMI category (health context) with the user's stated goal.
    goal_tips = {
        "Lose Weight": [
            "Keep a moderate calorie deficit (~500 kcal/day below TDEE) rather than crash dieting.",
            "Prioritize protein (lean meats, eggs, legumes, tofu, dairy) to preserve muscle while losing fat.",
            "Fill half your plate with vegetables and fiber-rich foods to stay full on fewer calories.",
            "Limit sugary drinks, fried foods, and refined carbs like white bread and pastries.",
            "Stay hydrated — thirst is often mistaken for hunger.",
        ],
        "Maintain": [
            "Match your intake to your TDEE and focus on balanced, whole-food meals.",
            "Include a mix of complex carbs (whole grains, oats, brown rice), lean protein, and healthy fats.",
            "Get a variety of fruits and vegetables daily for micronutrients and fiber.",
            "Practice portion awareness rather than strict restriction.",
            "Stay consistent with meal timing to support steady energy levels.",
        ],
        "Gain Muscle": [
            "Eat in a moderate surplus (~300-500 kcal above TDEE) to support muscle growth without excess fat gain.",
            "Aim for roughly 1.6-2.2 g of protein per kg of bodyweight, spread across meals.",
            "Prioritize complex carbs (rice, oats, potatoes, whole grains) around workouts for energy and recovery.",
            "Include healthy fats (nuts, seeds, olive oil, avocado) for hormone support.",
            "Pair nutrition with progressive resistance training and adequate sleep for best results.",
        ],
    }

    bmi_notes = {
        "Underweight": "Your BMI suggests you're underweight — a calorie surplus with nutrient-dense foods is generally advisable, and it's worth checking in with a healthcare provider if this is unintentional.",
        "Normal": "Your BMI is in the healthy range — the goal-based plan above should work well for you.",
        "Overweight": "Your BMI suggests you're in the overweight range — a gradual, sustainable deficit (if your goal is fat loss) tends to work better long-term than aggressive restriction.",
        "Obese": "Your BMI suggests you're in the obese range — consider consulting a healthcare provider or dietitian for a personalized plan alongside general guidance like this.",
    }

    st.markdown(
        f"""<div class="diet-card">
                <h4>📌 Based on your BMI ({bmi_category})</h4>
                <p>{bmi_notes[bmi_category]}</p>
            </div>""",
        unsafe_allow_html=True,
    )

    tips_html = "".join(f"<li>{tip}</li>" for tip in goal_tips[goal])
    st.markdown(
        f"""<div class="diet-card">
                <h4>🥗 Tips for your goal: {goal}</h4>
                <ul>{tips_html}</ul>
            </div>""",
        unsafe_allow_html=True,
    )

    st.caption(
        "⚠️ These figures are general estimates for informational purposes only, not medical advice. "
        "Consult a doctor or registered dietitian for guidance tailored to your health history."
    )

# =================================================
# Footer / extra info
# =================================================
st.divider()
with st.expander("About this app"):
    st.write("""
        This app uses a **Random Forest Regressor** model trained on workout data
        (age, height, weight, duration, heart rate, body temperature, BMI, and workout type)
        to estimate the number of calories burnt during a session, and pairs it with a
        Mifflin-St Jeor based daily calorie and macro-nutrient planner.

        - **Model:** RandomForestRegressor (scikit-learn)
        - **Test R² Score:** ~0.996
        - **Test MAE:** ~2.36 kcal
        - **BMR formula:** Mifflin-St Jeor equation
        - **TDEE:** BMR × activity multiplier

        Built with Streamlit.
    """)
