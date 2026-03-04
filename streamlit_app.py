import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Load the trained model and the feature columns
# Ensure these files are in the same directory as app.py
model = joblib.load('rf_churn_model.joblib')
model_columns = joblib.load('model_columns.pkl')

# 2. Page Title and Header
st.set_page_config(page_title="Churn Predictor", page_icon="📞")
st.title("📞 Telco Customer Churn Predictor")

# Informative description about the simplified model input
st.info("""
**Note:** This predictor focuses on the **Top 6 key features** of churn identified during my analysis. 
To ensure a streamlined user experience, all other technical features (such as Device Protection, 
Streaming TV, etc.) are set to their average values.
""")

st.write("---")

# 3. User Input Fields (Top Features)
st.subheader("Customer Profile")

# Using columns to organize the layout
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months with company)", 0, 72, 12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

with col2:
    monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0, 1000.0)
    internet_service = st.selectbox("Internet Service Type", ["Fiber optic", "DSL", "No"])

# 4. Data Preprocessing for Prediction
# Create a DataFrame with all columns initialized to 0
input_data = pd.DataFrame(0, index=[0], columns=model_columns)

# Map numeric inputs
input_data['tenure'] = tenure
input_data['MonthlyCharges'] = monthly_charges
input_data['TotalCharges'] = total_charges

# Map categorical inputs (Handling One-Hot Encoding manually)
# Contract logic
if contract == "One year":
    input_data['Contract_One year'] = 1
elif contract == "Two year":
    input_data['Contract_Two year'] = 1
# Note: Month-to-month stays 0 due to drop_first=True during encoding

# Internet Service logic
if internet_service == "Fiber optic":
    input_data['InternetService_Fiber optic'] = 1
elif internet_service == "No":
    input_data['InternetService_No'] = 1

# Paperless Billing logic
if paperless == "Yes":
    input_data['PaperlessBilling_Yes'] = 1

# 5. Prediction Logic
st.write("---")
if st.button("Run Churn Analysis"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.error(f"### ⚠️ High Risk of Churn")
        st.write(f"The model predicts a **{proba:.1%}** probability that this customer will leave.")
        st.markdown("---")
        st.write("**Retention Strategy Recommendations:**")
        st.write("* Offer a transition to a long-term contract (1 or 2 years).")
        st.write("* Provide a targeted discount on monthly charges.")
        st.write("* Check if the customer is experiencing technical issues with Fiber Optic service.")
    else:
        st.success(f"### ✅ Low Risk of Churn")
        st.write(f"The model predicts only a **{proba:.1%}** probability of churn.")
        st.markdown("---")
        st.write("**Growth Strategy Recommendations:**")
        st.write("* Customer is stable. Consider upselling premium bundles.")
        st.write("* Ideal candidate for loyalty program invitations.")

# Footer
st.caption("Developed as part of a michaeltsop.com Portfolio Project")