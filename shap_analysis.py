import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ==========================================
# Load Model
# ==========================================

model = joblib.load(
    "models/xgboost_model.pkl"
)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv(
    "data/final_train.csv"
)

# ==========================================
# Remove Target
# ==========================================

X = df.drop(columns=['failure_count'])

# ==========================================
# Keep Only Numeric Columns
# ==========================================

X = X.select_dtypes(
    include=['int64', 'float64']
)

# ==========================================
# Match Training Features
# ==========================================

X = X[model.feature_names_in_]

# ==========================================
# Sample Data for SHAP
# ==========================================

X_sample = X.sample(
    100,
    random_state=42
)

# ==========================================
# SHAP Explainer
# ==========================================

explainer = shap.Explainer(model)

# ==========================================
# Compute SHAP Values
# ==========================================

shap_values = explainer(X_sample)

# ==========================================
# Summary Plot
# ==========================================

shap.summary_plot(
    shap_values,
    X_sample
)

plt.show()