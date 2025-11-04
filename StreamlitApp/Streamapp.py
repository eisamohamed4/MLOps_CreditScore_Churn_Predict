{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ab18575b-2586-41da-b8ca-166921755bb8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import numpy as np\n",
    "import joblib\n",
    "import os\n",
    "\n",
    "# Load models, encoders, and scalers\n",
    "models_path = \"Models\"\n",
    "credit_model = joblib.load(os.path.join(models_path, \"credit_model.pkl\"))\n",
    "classification_model = joblib.load(os.path.join(models_path, \"classification_model.pkl\"))\n",
    "scaler = joblib.load(os.path.join(models_path, \"scaler.pkl\"))\n",
    "label_encoder_geography = joblib.load(os.path.join(models_path, \"label_encoder_geography.pkl\"))\n",
    "label_encoder_gender = joblib.load(os.path.join(models_path, \"label_encoder_gender.pkl\"))\n",
    "\n",
    "# Streamlit UI\n",
    "st.title(\"Customer Churn Prediction 🚀\")\n",
    "st.markdown(\"Enter customer details to predict **Credit Score** & whether they will leave the bank.\")\n",
    "\n",
    "# Input fields\n",
    "geography = st.selectbox(\"Geography\", [\"France\", \"Spain\", \"Germany\"])\n",
    "gender = st.selectbox(\"Gender\", [\"Male\", \"Female\"])\n",
    "age = st.number_input(\"Age\", min_value=18, max_value=100, value=35)\n",
    "tenure = st.number_input(\"Tenure (Years)\", min_value=0, max_value=10, value=5)\n",
    "balance = st.number_input(\"Balance ($)\", min_value=0.0, max_value=300000.0, value=75000.0)\n",
    "num_of_products = st.number_input(\"Number of Products\", min_value=1, max_value=4, value=2)\n",
    "has_cr_card = st.radio(\"Has Credit Card\", [1, 0])\n",
    "is_active_member = st.radio(\"Is Active Member\", [1, 0])\n",
    "\n",
    "# Predict button\n",
    "if st.button(\"Predict\"):\n",
    "    geography_encoded = label_encoder_geography.transform([geography])[0]\n",
    "    gender_encoded = label_encoder_gender.transform([gender])[0]\n",
    "\n",
    "    input_data = np.array([[geography_encoded, gender_encoded, age, tenure,\n",
    "                            balance, num_of_products, has_cr_card, is_active_member]])\n",
    "\n",
    "    input_scaled = scaler.transform(input_data)\n",
    "\n",
    "    predicted_credit_score_log = credit_model.predict(input_scaled)\n",
    "    predicted_credit_score = np.expm1(predicted_credit_score_log)\n",
    "\n",
    "    input_with_credit = np.hstack((input_scaled, predicted_credit_score.reshape(-1, 1)))\n",
    "\n",
    "    predicted_exit = classification_model.predict(input_with_credit)\n",
    "    exit_status = \"Will Leave\" if predicted_exit[0] == 1 else \"Will Stay\"\n",
    "\n",
    "    st.success(f\"Predicted Credit Score: **{predicted_credit_score[0]:.2f}**\")\n",
    "    st.warning(f\"Customer Prediction: **{exit_status}**\" if exit_status == \"Will Leave\" else f\"✅ {exit_status}\")\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
