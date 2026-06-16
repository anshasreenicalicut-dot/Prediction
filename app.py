from flask import Flask, request, jsonify
import pandas as pd
import joblib
from datetime import datetime

# =====================================
# Load Model
# =====================================

model = joblib.load("C:\\Users\\Ansha TV\\Desktop\\AzurePredictiveMaintenance\\models\\xgboost_model.pkl")

feature_columns = joblib.load(
    "C:\\Users\\Ansha TV\\Desktop\\AzurePredictiveMaintenance\\models\\feature_columns.pkl"
)

app = Flask(__name__)

# =====================================
# Home Route
# =====================================

@app.route("/")
def home():

    return jsonify({
        "message":
        "Azure Predictive Maintenance API Running"
    })

# =====================================
# Prediction Route
# =====================================

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    volt = data["volt"]
    rotate = data["rotate"]
    pressure = data["pressure"]
    vibration = data["vibration"]

    machineID = data.get("machineID", 1)
    maintenance_count = data.get(
        "maintenance_count",
        0
    )

    now = datetime.now()

    input_data = {

        "machineID": machineID,

        "volt": volt,
        "rotate": rotate,
        "pressure": pressure,
        "vibration": vibration,

        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "day_of_week": now.weekday(),

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

    df = pd.DataFrame([input_data])

    for col in feature_columns:

        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    prediction = model.predict(df)[0]

    probability = float(
        model.predict_proba(df)[0][1]
    )

    return jsonify({

        "prediction":
        int(prediction),

        "failure_probability":
        round(probability, 4),

        "status":
        "Failure Predicted"
        if prediction == 1
        else "Machine Healthy"
    })

# =====================================
# Run
# =====================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )