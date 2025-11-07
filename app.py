import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load("model_log.pkl")    



st.title("🍄Breast Cancer Prediction App")   

st.write("Enter the tumor characteristics below to predict if it's **Benign** or **Malignant**.")






f1 = st.number_input("Mean Radius: ", format = "%.4f")
f2 = st.number_input("Texture Mean: ", format = "%.4f")
f3 = st.number_input("Perimeter Mean: ", format = "%.4f")
f4 = st.number_input("Area Mean: ", format = "%.4f")
f5 = st.number_input("Smoothness Mean: ", format = "%.4f")
f6 = st.number_input("Compactness Mean: ", format = "%.4f")
f7 = st.number_input("Concavity Mean: ", format = "%.4f")
f8 = st.number_input("Concave Points Mean: ", format = "%.4f")
f9 = st.number_input("Symmetry Mean: ", format = "%.4f")
f10 = st.number_input("Fractal Dimension Mean: ", format = "%.4f")



input_data = np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]])



if st.button("🔍Predict"):
    pred = model.predict(input_data)
    prob = model.predict_proba(input_data)

    result_label = "🩸 Malignant" if pred[0]==1 else "✅ Benign!"

    st.success(f"Diagnosis: **{result_label}**")

    st.info(f"Probability for Tumor: {prob[0][1]:.2f}")