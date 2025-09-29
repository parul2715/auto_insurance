import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# --- Custom KPI Block ---
def custom_kpi(label, value, value_color="#ffffff", bg_color="#1e1e1e"):
    st.markdown(f"""
        <div style="padding: 15px 20px; border-radius: 12px; background-color: {bg_color};
                    border: 1px solid #333; margin-bottom: 10px; text-align:center;">
            <div style="font-size: 14px; font-weight: 500; color: #cccccc;">{label}</div>
            <div style="font-size: 22px; font-weight: bold; color: {value_color};">{value}</div>
        </div>
    """, unsafe_allow_html=True)

# --- Helper for inputs ---
def typed_number_input(label, value=0, step=1, is_int=False):
    if is_int:
        return st.number_input(label, value=value, step=step)
    else:
        return st.number_input(label, value=float(value))

# --- Load models, encoders, scaler ---
rf_model = joblib.load('random_forest_best_model.pkl')
baseline_model = joblib.load('baseline_logistic_regression.pkl')
encoders = joblib.load('encoders.pkl')
scaler = joblib.load('scaler.pkl')

# Try loading model comparison results
try:
    comparison = joblib.load("model_comparison.pkl")
except:
    comparison = None

st.title("🚗 Auto Insurance Response Prediction")

# --- Sidebar for inputs ---
with st.sidebar:
    st.image("image.webp", use_container_width=True)
    st.header("Input Features")

    # --- Model choice ---
    model_choice = st.radio("Select Model", ["Random Forest", "Logistic Regression"])
    model = rf_model if model_choice == "Random Forest" else baseline_model
    explainer = shap.TreeExplainer(model) if model_choice == "Random Forest" else None

    # --- Inputs ---
    income = typed_number_input("Income", value=43072)
    monthly_premium_auto = typed_number_input("Monthly Premium Auto", value=159)
    months_since_last_claim = typed_number_input("Months Since Last Claim", value=24, step=1, is_int=True)
    months_since_policy_inception = typed_number_input("Months Since Policy Inception", value=54, step=1, is_int=True)
    number_of_open_complaints = typed_number_input("Number of Open Complaints", value=3, step=1, is_int=True)
    number_of_policies = typed_number_input("Number of Policies", value=4, step=1, is_int=True)
    total_claim_amount = typed_number_input("Total Claim Amount", value=350)
    clv_corrected = typed_number_input("CLV Corrected", value=6000)  # 🔹 renamed for consistency
    effective_to_date = st.date_input("Effective To Date")

    def get_options(col):
        return list(encoders[col].classes_) if col in encoders else []

    coverage = st.selectbox("Coverage", get_options('Coverage'))
    education = st.selectbox("Education", get_options('Education'))
    employment_status = st.selectbox("EmploymentStatus", get_options('EmploymentStatus'))
    gender = st.selectbox("Gender", get_options('Gender'))
    location_code = st.selectbox("Location_Code", get_options('Location_Code'))
    marital_status = st.selectbox("Marital_Status", get_options('Marital_Status'))
    policy_type = st.selectbox("Policy_Type", get_options('Policy_Type'))
    policy = st.selectbox("Policy", get_options('Policy'))
    renew_offer_type = st.selectbox("Renew_Offer_Type", get_options('Renew_Offer_Type'))
    sales_channel = st.selectbox("Sales_Channel", get_options('Sales_Channel'))
    vehicle_class = st.selectbox("Vehicle_Class", get_options('Vehicle_Class'))
    vehicle_size = st.selectbox("Vehicle_Size", get_options('Vehicle_Size'))
    state = st.selectbox("State", get_options('State'))

    predict_button = st.button("Predict Response")

# --- Prepare Input DataFrame ---
input_dict = {
    'Income': [income],
    'Monthly_Premium_Auto': [monthly_premium_auto],
    'Months_Since_Last_Claim': [months_since_last_claim],
    'Months_Since_Policy_Inception': [months_since_policy_inception],
    'Number_of_Open_Complaints': [number_of_open_complaints],
    'Number_of_Policies': [number_of_policies],
    'Total_Claim_Amount': [total_claim_amount],
    'CLV_Corrected': [clv_corrected],
    'Coverage': [coverage],
    'Education': [education],
    'EmploymentStatus': [employment_status],
    'Gender': [gender],
    'Location_Code': [location_code],
    'Marital_Status': [marital_status],
    'Policy_Type': [policy_type],
    'Policy': [policy],
    'Renew_Offer_Type': [renew_offer_type],
    'Sales_Channel': [sales_channel],
    'Vehicle_Class': [vehicle_class],
    'Vehicle_Size': [vehicle_size],
    'State': [state],
    'Effective_To_Date': [effective_to_date.strftime('%Y-%m-%d')]
}

input_df = pd.DataFrame(input_dict)

# Encode categorical
for col in encoders:
    if col in input_df.columns:
        input_df[col] = encoders[col].transform(input_df[col].astype(str))

# Convert date
input_df['Effective_To_Date'] = pd.to_datetime(input_df['Effective_To_Date'], errors='coerce')
input_df['Effective_To_Date'] = input_df['Effective_To_Date'].astype('int64') // 10 ** 9

# Scale numeric
numeric_cols = list(scaler.feature_names_in_)
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# Ensure correct column order
input_df = input_df[model.feature_names_in_]

# --- Prediction ---
if predict_button:
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0, 1]

    st.header("Model Prediction")

    col1, col2 = st.columns(2)

    with col1:
        response_text = "Yes" if pred == 1 else "No"
        response_color = "#00FF00" if response_text == "Yes" else "#FF6347"
        custom_kpi("Response", response_text, value_color=response_color)

    with col2:
        custom_kpi("Probability", f"{proba:.2%}", value_color="#00BFFF")

    # --- SHAP Explanation (only for RF) ---
    if explainer:
        shap_values = explainer.shap_values(input_df)
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[0]

        shap_values_for_instance = shap_values
        if isinstance(shap_values, (list, np.ndarray)):
            shap_values_for_instance = shap_values[0]

        if hasattr(shap_values_for_instance, "ndim") and shap_values_for_instance.ndim > 1:
            shap_values_for_instance = shap_values_for_instance[0]

        st.header("Prediction Explanation (SHAP Waterfall Plot)")
        fig, ax = plt.subplots()
        shap.plots._waterfall.waterfall_legacy(
            base_value,
            shap_values_for_instance,
            max_display=10,
            feature_names=input_df.columns.tolist(),
            show=False)
        st.pyplot(fig)

        st.header("Top Features Influencing Prediction")
        abs_shap = np.abs(shap_values_for_instance)
        top_indices = np.argsort(abs_shap)[-5:][::-1]
        for i in top_indices:
            f_name = input_df.columns[i]
            f_val = input_df.iloc[0, i]
            contribution = shap_values_for_instance[i]
            st.write(f"**{f_name}** (value: {f_val}) → impact: {contribution:.4f}")
    else:
        st.info("SHAP explanation available only for Random Forest model.")


else:
    st.info("Enter input features in the sidebar and click Predict Response.")