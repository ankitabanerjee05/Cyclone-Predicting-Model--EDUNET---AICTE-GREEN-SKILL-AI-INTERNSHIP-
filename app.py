import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# Train Model
@st.cache_resource
def train_model():
    file = pd.read_csv("cyclone_occurrence_dataset.csv")

    X = file.drop('Cyclone', axis=1)
    y = file['Cyclone']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_scaled, y)

    return rf, scaler

rf, scaler = train_model()
st.set_page_config(page_title="🌪️ Cyclone Prediction App", layout="wide")


# Custom CSS Styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1D0826;  /* Dark navy background */
    }
    h1 {
        color: #FFDEE9;  /* soft pink for contrast */
        text-align: center;
    }
    .stButton>button {
        background-color: #7b2cbf;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #9d4edd;
        color: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("🚨 Cyclone Prediction System 🚨")
st.markdown("### Enter environmental parameters to check **cyclone occurrence chances**.")

col1, col2 = st.columns(2)
with col1:
    rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, max_value=1000.0, value=00.0, step=1.0)
    duration = st.number_input("⏱️ Duration (days)", min_value=0.0, max_value=100.0, value=00.0, step=1.0)
    humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=00.0, step=1.0)

with col2:
    sea_surface_temp = st.number_input("🌊 Sea Surface Temperature (°C)", min_value=00.0, max_value=100.0, value=00.0, step=0.1)
    wind_speed = st.number_input("🌬️ Wind Speed (knots)", min_value=0.0, max_value=1000.0, value=00.0, step=1.0)
    pressure = st.number_input("🌡️ Atmospheric Pressure (hPa)", min_value=0.0, max_value=10050.0, value=00.0, step=1.0)

st.markdown("---")

if st.button("🔍 Predict Cyclone"):
    user_data = np.array([[rainfall, duration, humidity, sea_surface_temp, wind_speed, pressure]])
    user_data_scaled = scaler.transform(user_data)

    prediction = rf.predict(user_data_scaled)
    probability = rf.predict_proba(user_data_scaled)[0][1] * 100

    if prediction[0] == 1:
        st.error(f"⚠️ Cyclone Predicted! **(Probability: {probability:.2f}%)**")
    else:
        st.success(f"✅ No Cyclone Predicted. **(Probability: {probability:.2f}%)**")

