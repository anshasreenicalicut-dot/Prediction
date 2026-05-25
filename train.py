import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from xgboost import XGBClassifier

# =========================
# Paths
# =========================

BASE_DIR = Path(r"C:\Users\Ansha TV\Desktop\AzurePredictiveMaintenance\data").resolve()
DATA_DIR = BASE_DIR
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Load datasets
# =========================

telemetry = pd.read_csv(DATA_DIR / "C:\\Users\\Ansha TV\\Desktop\\AzurePredictiveMaintenance\\data\\PdM_telemetry.csv")
failures = pd.read_csv(DATA_DIR / "C:\\Users\\Ansha TV\\Desktop\\AzurePredictiveMaintenance\\data\\PdM_failures.csv")
machines = pd.read_csv(DATA_DIR / "C:\\Users\\Ansha TV\\Desktop\\AzurePredictiveMaintenance\\data\\PdM_machines.csv")

# =========================
# Convert datetime
# =========================

telemetry['datetime'] = pd.to_datetime(telemetry['datetime'])

failures['datetime'] = pd.to_datetime(failures['datetime'])

# =========================
# Create failure label
# =========================

telemetry['failure'] = 0

failure_times = failures[['machineID', 'datetime']]

for idx, row in failure_times.iterrows():

    machine = row['machineID']
    fail_time = row['datetime']

    telemetry.loc[
        (
            (telemetry['machineID'] == machine)
            &
            (telemetry['datetime'] <= fail_time)
        ),
        'failure'
    ] = 1

# =========================
# Merge machine metadata
# =========================

df = telemetry.merge(
    machines,
    on='machineID',
    how='left'
)

# =========================
# Encode categorical columns
# =========================

encoder = LabelEncoder()

df['model'] = encoder.fit_transform(df['model'])

# =========================
# Drop datetime
# =========================

df.drop(['datetime'], axis=1, inplace=True)

# =========================
# Features & Target
# =========================

X = df.drop('failure', axis=1)

y = df['failure']

# =========================
# Train-Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# Model
# =========================

model = XGBClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

# =========================
# Train
# =========================

model.fit(X_train, y_train)

# =========================
# Predict
# =========================

y_pred = model.predict(X_test)

# =========================
# Evaluation
# =========================

print(classification_report(y_test, y_pred))

# =========================
# Save model
# =========================

from pathlib import Path

# Create folder
Path("models").mkdir(exist_ok=True)

# Save model
joblib.dump(
    model,
    "models/xgboost_model.pkl"
)

print("Model saved successfully")