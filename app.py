import streamlit as st
import pickle
import numpy as np

st.title("Diabetes Prediction App")
st.write("Upload your trained model and enter patient details to predict diabetes.")

# Upload the trained model file
uploaded_file = st.file_uploader("Upload Diabetes_Prediction.pkl", type=["pkl"])

if uploaded_file is not None:
    # Load the model
    model = pickle.load(uploaded_file)
    
    st.subheader("Enter Patient Details:")
    age = st.number_input("Age:", min_value=1, max_value=120, value=30)
    mass = st.number_input("Mass (BMI):", min_value=1, max_value=100, value=25)
    insulin = st.number_input("Insulin Level:", min_value=0, max_value=1000, value=100)
    plasma = st.number_input("Plasma Level:", min_value=0, max_value=300, value=120)

    # Prediction
    if st.button("Predict"):
        pred = model.predict(np.array([[age, mass, insulin, plasma]]))
        if pred[0] == 1:
            st.error("The patient is likely to have diabetes.")
        else:
            st.success("The patient is not likely to have diabetes.")
else:
    st.warning("Please upload the trained model file (.pkl) to continue.")
