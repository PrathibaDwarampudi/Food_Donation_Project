import streamlit as st
import numpy as np

st.title("AI-Based Food Donation Need Predictor")

st.write("This system predicts food donation need based on area conditions.")

# User inputs
population = st.number_input("Population Density", min_value=0)
income = st.number_input("Average Income", min_value=0)
poverty = st.slider("Poverty Rate (%)", 0, 100)
unemployment = st.slider("Unemployment Rate (%)", 0, 100)
event = st.selectbox("Event Type", ["None", "Festival", "Disaster"])
surplus = st.number_input("Surplus Food (kg)", min_value=0)

# Simple AI logic (rule-based prediction)
if st.button("Predict"):
    score = 0

    if poverty > 40:
        score += 2
    elif poverty > 20:
        score += 1

    if unemployment > 15:
        score += 2
    elif unemployment > 8:
        score += 1

    if event == "Disaster":
        score += 2
    elif event == "Festival":
        score += 1

    if surplus < 150:
        score += 2
    elif surplus < 300:
        score += 1

    if score >= 6:
        st.error("High Food Donation Need")
    elif score >= 3:
        st.warning("Medium Food Donation Need")
    else:
        st.success("Low Food Donation Need")
