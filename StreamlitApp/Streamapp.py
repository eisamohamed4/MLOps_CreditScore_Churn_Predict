#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import joblib
import os

# Load models, encoders, and scalers
models_path = "Models"
credit_model = joblib.load(os.path.join(models_path, "credit_model.pkl"))
classification_model = joblib.load(os.path.join(models_path, "classification_model.pkl"))
scaler = joblib.load(os.path.join(models_path, "scaler.pkl"))
label_encoder_geography = joblib.load(os.path.join(models_path, "label_encoder_geography.pkl"))
label_encoder_gender = joblib.load(os.path.join(models_path, "label_encoder_gender.pkl"))

# Streamlit UI
st.title("Customer Churn Prediction 🚀")
st.markdown("Enter customer details to predict **Credit Score** & whether they will leave the bank.")

# Input fields
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=35)
tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)
balance = st.number_input("Balance ($)", min_value=0.0, max_value=300000.0, value=75000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
has_cr_card = st.radio("Has Credit Card", [1, 0])
is_active_member = st.radio("Is Active Member", [1, 0])

# Predict button
if st.button("Predict"):
    geography_encoded = label_encoder_geography.transform([geography])[0]
    gender_encoded = label_encoder_gender.transform([gender])[0]

    input_data = np.array([[geography_encoded, gender_encoded, age, tenure,
                            balance, num_of_products, has_cr_card, is_active_member]])

    input_scaled = scaler.transform(input_data)

    predicted_credit_score_log = credit_model.predict(input_scaled)
    predicted_credit_score = np.expm1(predicted_credit_score_log)

    input_with_credit = np.hstack((input_scaled, predicted_credit_score.reshape(-1, 1)))

    predicted_exit = classification_model.predict(input_with_credit)
    exit_status = "Will Leave" if predicted_exit[0] == 1 else "Will Stay"

    st.success(f"Predicted Credit Score: **{predicted_credit_score[0]:.2f}**")
    st.warning(f"Customer Prediction: **{exit_status}**" if exit_status == "Will Leave" else f"✅ {exit_status}")


