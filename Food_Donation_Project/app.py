import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("AI-Based Food Donation Need Predictor")

# Load dataset
df = pd.read_csv("data.csv")

# Convert event type to numbers
df["event_type"] = df["event_type"].astype(str).str.lower().map({
    "none": 0,
    "festival": 1,
    "disaster": 2
})

# Split features and target
X = df.drop("donation_need", axis=1)
y = df["donation_need"]

# Train model inside app
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# User input
population = st.number_input("Population Density", min_value=0)
income = st.number_input("Average Income", min_value=0)
poverty = st.slider("Poverty Rate (%)", 0, 100)
unemployment = st.slider("Unemployment Rate (%)", 0, 100)
event = st.selectbox("Event Type", ["None", "Festival", "Disaster"])
surplus = st.number_input("Surplus Food (kg)", min_value=0)

event_map = {"None": 0, "Festival": 1, "Disaster": 2}

# Prediction
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
