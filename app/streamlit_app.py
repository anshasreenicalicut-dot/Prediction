# =====================================================
# Azure Predictive Maintenance Dashboard
# =====================================================

import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Azure Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.metric-card {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "C:\\Users\\Ansha TV\\Desktop\\AzurePredictiveMaintenance\\models\\xgboost_model.pkl"
    )

    feature_columns = joblib.load(
        "C:\\Users\\Ansha TV\\Desktop\\AzurePredictiveMaintenance\\models\\feature_columns.pkl"
    )

    return model, feature_columns


try:
    model, feature_columns = load_model()

except Exception as e:

    st.error(
        f"Model Loading Failed:\n\n{e}"
    )

    st.stop()

# =====================================================
# HEADER
# =====================================================

st.title("⚙️ Azure Predictive Maintenance")

st.markdown("""
Real-time machine health monitoring using
**XGBoost Machine Learning** and advanced
telemetry feature engineering.
""")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Machine Configuration")

machineID = st.sidebar.number_input(
    "Machine ID",
    min_value=1,
    max_value=1000,
    value=1
)

maintenance_count = st.sidebar.number_input(
    "Maintenance Count",
    min_value=0,
    max_value=50,
    value=0
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Adjust sensor readings and click "
    "'Predict Failure Risk'."
)

# =====================================================
# SENSOR INPUTS
# =====================================================

st.subheader("Telemetry Sensor Readings")

col1, col2 = st.columns(2)

with col1:

    volt = st.slider(
        "Voltage",
        100,
        300,
        170
    )

    rotate = st.slider(
        "Rotation Speed",
        0,
        500,
        450
    )

with col2:

    pressure = st.slider(
        "Pressure",
        0,
        200,
        110
    )

    vibration = st.slider(
        "Vibration",
        0,
        100,
        50
    )

# =====================================================
# KPI CARDS
# =====================================================

st.subheader("Current Sensor Status")

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Voltage",
    volt
)

k2.metric(
    "Rotation",
    rotate
)

k3.metric(
    "Pressure",
    pressure
)

k4.metric(
    "Vibration",
    vibration
)

# =====================================================
# TIME FEATURES
# =====================================================

now = datetime.now()

hour = now.hour
day = now.day
month = now.month
day_of_week = now.weekday()

# =====================================================
# CREATE INPUT DATA
# =====================================================

input_data = {

    "machineID": machineID,

    "volt": volt,
    "rotate": rotate,
    "pressure": pressure,
    "vibration": vibration,

    "hour": hour,
    "day": day,
    "month": month,
    "day_of_week": day_of_week,

    "volt_lag_1": volt,
    "volt_lag_2": volt,

    "rotate_lag_1": rotate,
    "rotate_lag_2": rotate,

    "pressure_lag_1": pressure,
    "pressure_lag_2": pressure,

    "vibration_lag_1": vibration,
    "vibration_lag_2": vibration,

    "volt_rolling_mean_3": volt,
    "volt_rolling_std_3": 0,

    "volt_rolling_mean_6": volt,
    "volt_rolling_std_6": 0,

    "volt_rolling_mean_12": volt,
    "volt_rolling_std_12": 0,

    "rotate_rolling_mean_3": rotate,
    "rotate_rolling_std_3": 0,

    "rotate_rolling_mean_6": rotate,
    "rotate_rolling_std_6": 0,

    "rotate_rolling_mean_12": rotate,
    "rotate_rolling_std_12": 0,

    "pressure_rolling_mean_3": pressure,
    "pressure_rolling_std_3": 0,

    "pressure_rolling_mean_6": pressure,
    "pressure_rolling_std_6": 0,

    "pressure_rolling_mean_12": pressure,
    "pressure_rolling_std_12": 0,

    "vibration_rolling_mean_3": vibration,
    "vibration_rolling_std_3": 0,

    "vibration_rolling_mean_6": vibration,
    "vibration_rolling_std_6": 0,

    "vibration_rolling_mean_12": vibration,
    "vibration_rolling_std_12": 0,

    "volt_ema_5": volt,
    "rotate_ema_5": rotate,
    "pressure_ema_5": pressure,
    "vibration_ema_5": vibration,

    "volt_std_from_mean": 0,
    "rotate_std_from_mean": 0,
    "pressure_std_from_mean": 0,
    "vibration_std_from_mean": 0,

    "power": volt * rotate,

    "stress": pressure * vibration,

    "maintenance_count": maintenance_count
}

input_df = pd.DataFrame([input_data])

# =====================================================
# FEATURE ALIGNMENT
# =====================================================

for col in feature_columns:

    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[feature_columns]

# =====================================================
# PREDICTION BUTTON
# =====================================================

if st.button(
    "🚀 Predict Failure Risk",
    use_container_width=True
):

    prediction = model.predict(
        input_df
    )[0]

    probability = model.predict_proba(
        input_df
    )[0][1]

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            f"⚠️ HIGH FAILURE RISK\n\n"
            f"Probability: {probability:.2%}"
        )

    else:

        st.success(
            f"✅ MACHINE HEALTHY\n\n"
            f"Confidence: {(1-probability):.2%}"
        )

    # =========================================
    # RISK METER
    # =========================================

    st.subheader("Failure Probability")

    st.progress(float(probability))

    st.metric(
        "Risk Score",
        f"{probability:.2%}"
    )

    # =========================================
    # SENSOR TABLE
    # =========================================

    st.subheader("Machine Summary")

    summary = pd.DataFrame({

        "Feature": [
            "Machine ID",
            "Voltage",
            "Rotation",
            "Pressure",
            "Vibration",
            "Maintenance Count"
        ],

        "Value": [
            machineID,
            volt,
            rotate,
            pressure,
            vibration,
            maintenance_count
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True
    )

    # =========================================
    # MODEL DETAILS
    # =========================================

    st.subheader("Model Information")

    st.info(
        """
        Model : XGBoost Classifier
        
        Objective : Predict Machine Failure
        
        Evaluation Metric : Precision-Recall AUC
        
        Imbalance Handling : SMOTE
        
        Feature Engineering :
        Lag Features, Rolling Statistics,
        EMA Features, Interaction Features
        """
    )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Azure Predictive Maintenance Platform | "
    "Powered by XGBoost, Streamlit & Azure ML"
)