import streamlit as st
import pickle
import numpy as np
import os

MODEL_PATH = "Diabetes_Prediction.pkl"

# Check if model exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}. Upload it in the same folder as app.py.")
    st.stop()

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

st.title("Diabetes Prediction App")

age = st.number_input("Enter Age:", min_value=1, max_value=120, value=30)
mass = st.number_input("Enter Mass (BMI):", min_value=1, max_value=100, value=25)
insulin = st.number_input("Enter Insulin Level:", min_value=0, max_value=1000, value=100)
plasma = st.number_input("Enter Plasma Level:", min_value=0, max_value=300, value=120)

if st.button("Predict"):
    pred = model.predict(np.array([[age, mass, insulin, plasma]]))
    if pred[0] == 1:
        st.error("The patient is likely to have diabetes.")
    else:
        st.success("The patient is not likely to have diabetes.")
