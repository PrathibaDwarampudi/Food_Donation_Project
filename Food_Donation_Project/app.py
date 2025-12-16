import streamlit as st
import pickle
import numpy as np

import os
import pickle

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = pickle.load(open(MODEL_PATH, "rb"))


st.title("Food Shortage Risk Predictor")

population = st.number_input("Population Density")
income = st.number_input("Average Income")
poverty = st.slider("Poverty Rate", 0, 100)
unemployment = st.slider("Unemployment Rate", 0, 100)
event = st.selectbox("Event Type", ["None", "Festival", "Disaster"])
surplus = st.number_input("Surplus Food (kg)")

event_map = {"None": 0, "Festival": 1, "Disaster": 2}

if st.button("Predict"):
    data = np.array([[population, income, poverty, unemployment, event_map[event], surplus]])
    result = model.predict(data)

    if result[0] == 2:
        st.error("High Food Shortage Risk")
    elif result[0] == 1:
        st.warning("Medium Food Shortage Risk")
    else:
        st.success("Low Food Shortage Risk")

