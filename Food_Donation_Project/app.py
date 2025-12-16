import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

# -------------------------------
# Title
# -------------------------------
st.title("AI-Based Food Donation Need Predictor")

# -------------------------------
# Load CSV safely
# -------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data.csv")
df = pd.read_csv(DATA_PATH)

# -------------------------------
# Force numeric conversion (KEY FIX)
# -------------------------------
# Convert event_type text to numbers
df["event_type"] = (
    df["event_type"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "none": 0,
        "festival": 1,
        "disaster": 2
    })
)

# Convert ALL columns to numeric (very important)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with ANY missing or invalid values
df = df.dropna()

# -------------------------------
# Split features and target
# -------------------------------
X = df.drop("donation_need", axis=1)
y = df["donation_need"]

# -------------------------------
# Train model
# -------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# -------------------------------
# User Inputs
# -------------------------------
population = st.number_input("Population Density", min_value=0)
income = st.number_input("Average Income", min_value=0)
poverty = st.slider("Poverty Rate (%)", 0, 100)
unemployment = st.slider("Unemployment Rate (%)", 0, 100)
event = st.selectbox("Event Type", ["None", "Festival", "Disaster"])
surplus = st.number_input("Surplus Food (kg)", min_value=0)

event_map = {"None": 0, "Festival": 1, "Disaster": 2}

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    input_data = np.array([[population, income, poverty,
                             unemployment, event_map[event], surplus]])

    prediction = model.predict(input_data)[0]

    if prediction == 2:
        st.error("High Food Donation Need")
    elif prediction == 1:
        st.warning("Medium Food Donation Need")
    else:
        st.success("Low Food Donation Need")
