import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('Diabetes_Prediction.pkl', 'rb') as f:
    model = pickle.load(f)

# Title
st.title("Diabetes Prediction App")
st.write("Enter the patient's details to predict diabetes.")

# User input
age = st.number_input("Enter Age:", min_value=1, max_value=120, value=30)
mass = st.number_input("Enter Mass (BMI):", min_value=1, max_value=100, value=25)
insulin = st.number_input("Enter Insulin Level:", min_value=0, max_value=1000, value=100)
plasma = st.number_input("Enter Plasma Level:", min_value=0, max_value=300, value=120)

# Prediction button
if st.button("Predict"):
    # Make prediction
    pred = model.predict(np.array([[age, mass, insulin, plasma]]))
    if pred[0] == 1:
        st.error("The patient is likely to have diabetes.")
    else:
        st.success("The patient is not likely to have diabetes.")

# Model Accuracy
if st.checkbox("Show model accuracy"):
    st.write(f"Model training accuracy: {model.score(model._validate_data, model._validate_targets):.2f}")
