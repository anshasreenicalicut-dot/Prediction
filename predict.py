import pandas as pd
import joblib

# ==========================================
# Load Model
# ==========================================

model = joblib.load(
    "models/xgboost_model.pkl"
)

# ==========================================
# Sample Input
# ==========================================

volt = 170
rotate = 450
pressure = 110
vibration = 50

# ==========================================
# Create Input Data
# ==========================================

sample = pd.DataFrame({

    'machineID': [1],

    'volt': [volt],
    'rotate': [rotate],
    'pressure': [pressure],
    'vibration': [vibration],

    # Time Features
    'hour': [10],
    'day': [15],
    'month': [5],
    'day_of_week': [2],

    # Lag Features
    'volt_lag_1': [volt],
    'volt_lag_2': [volt],

    'rotate_lag_1': [rotate],
    'rotate_lag_2': [rotate],

    'pressure_lag_1': [pressure],
    'pressure_lag_2': [pressure],

    'vibration_lag_1': [vibration],
    'vibration_lag_2': [vibration],

    # Rolling Mean Features
    'volt_rolling_mean_3': [volt],
    'volt_rolling_std_3': [0],

    'volt_rolling_mean_6': [volt],
    'volt_rolling_std_6': [0],

    'volt_rolling_mean_12': [volt],
    'volt_rolling_std_12': [0],

    'rotate_rolling_mean_3': [rotate],
    'rotate_rolling_std_3': [0],

    'rotate_rolling_mean_6': [rotate],
    'rotate_rolling_std_6': [0],

    'rotate_rolling_mean_12': [rotate],
    'rotate_rolling_std_12': [0],

    'pressure_rolling_mean_3': [pressure],
    'pressure_rolling_std_3': [0],

    'pressure_rolling_mean_6': [pressure],
    'pressure_rolling_std_6': [0],

    'pressure_rolling_mean_12': [pressure],
    'pressure_rolling_std_12': [0],

    'vibration_rolling_mean_3': [vibration],
    'vibration_rolling_std_3': [0],

    'vibration_rolling_mean_6': [vibration],
    'vibration_rolling_std_6': [0],

    'vibration_rolling_mean_12': [vibration],
    'vibration_rolling_std_12': [0],

    # EMA Features
    'volt_ema_5': [volt],
    'rotate_ema_5': [rotate],
    'pressure_ema_5': [pressure],
    'vibration_ema_5': [vibration],

    # Interaction Features
    'power': [volt * rotate],
    'stress': [pressure * vibration],

    # Maintenance Feature
    'maintenance_count': [0]
})

# ==========================================
# Prediction
# ==========================================

prediction = model.predict(sample)

# ==========================================
# Output
# ==========================================

if prediction[0] == 1:
    print("⚠️ Failure Predicted")
else:
    print("✅ Machine Healthy")