import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load("model_log.pkl")    
    pred = model.predict(input_data)
    prob = model.predict_proba(input_data)

    result_label = "🩸 Malignant" if pred[0]==1 else "✅ Benign!"

    st.success(f"Diagnosis: **{result_label}**")

    st.info(f"Probability for Tumor: {prob[0][1]:.2f}")