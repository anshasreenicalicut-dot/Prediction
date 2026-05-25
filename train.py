# ==========================================
# Train XGBoost with GridSearchCV
# ==========================================

import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from xgboost import XGBClassifier

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("data/final_train.csv")

# ==========================================
# Target Variable
# ==========================================

target = "failure_count"

# ==========================================
# Features and Labels
# ==========================================

X = df.drop(columns=[target])

# Keep only numeric columns
X = X.select_dtypes(
    include=['int64', 'float64']
)
 

y = (df[target] > 0).astype(int)
# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
)

# ==========================================
# Base Model
# ==========================================

xgb_model = XGBClassifier(
    eval_metric='logloss',
    random_state=42
)

# ==========================================
# Hyperparameter Grid
# ==========================================

param_grid = {

    'n_estimators': [100, 200],

    'max_depth': [4, 6, 8],

    'learning_rate': [0.01, 0.05, 0.1],

    'subsample': [0.8, 1.0],

    'colsample_bytree': [0.8, 1.0]
}

# ==========================================
# GridSearchCV
# ==========================================

grid_search = GridSearchCV(

    estimator=xgb_model,

    param_grid=param_grid,

    scoring='accuracy',

    cv=3,

    verbose=2,

    n_jobs=-1
)

# ==========================================
# Train Model
# ==========================================

grid_search.fit(
    X_train,
    y_train
)

# ==========================================
# Best Model
# ==========================================

best_model = grid_search.best_estimator_

print("\nBest Parameters:")

print(grid_search.best_params_)

# ==========================================
# Predictions
# ==========================================

y_pred = best_model.predict(X_test)

# ==========================================
# Evaluation
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ==========================================
# Save Best Model
# ==========================================
import os

# Create models folder
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(
    best_model,
    "models/xgboost_model.pkl"
)

print("Model saved successfully!")
# ==========================================
