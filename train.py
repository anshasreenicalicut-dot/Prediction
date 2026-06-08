import os
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ==========================================
# LOAD DATA
# ==========================================

print("Loading dataset...")

df = pd.read_csv("data/final_train.csv")

print("Dataset Shape:", df.shape)

# ==========================================
# TARGET COLUMN
# ==========================================

TARGET = "failure_count"

if TARGET not in df.columns:
    print("\nAvailable Columns:")
    print(df.columns.tolist())
    raise ValueError(f"{TARGET} column not found!")

# ==========================================
# FEATURES & LABELS
# ==========================================

X = df.drop(columns=[TARGET])

# Remove datetime column if present
if "datetime" in X.columns:
    X = X.drop(columns=["datetime"])

# Keep only numeric columns
X = X.select_dtypes(include=["int64", "float64"])

y = df[TARGET]

# Convert to binary classification
y = (y > 0).astype(int)

print("\nTarget Distribution:")
print(y.value_counts())

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# SMOTE
# ==========================================

print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# ==========================================
# MODEL
# ==========================================

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

# ==========================================
# SMALL GRID SEARCH
# ==========================================

param_grid = {

    "n_estimators": [100, 200],

    "max_depth": [4, 6],

    "learning_rate": [0.1]

}

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring="average_precision",
    cv=3,
    verbose=2,
    n_jobs=-1
)

print("\nRunning GridSearchCV...")

grid_search.fit(
    X_train_smote,
    y_train_smote
)

# ==========================================
# BEST MODEL
# ==========================================

best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = best_model.predict(X_test)

y_prob = best_model.predict_proba(
    X_test
)[:, 1]

# ==========================================
# METRICS
# ==========================================

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

pr_auc = average_precision_score(
    y_test,
    y_prob
)

print("\n========== RESULTS ==========")

print("Precision :", round(precision, 4))
print("Recall    :", round(recall, 4))
print("F1 Score  :", round(f1, 4))
print("PR-AUC    :", round(pr_auc, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    y_pred
))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred
))

# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    best_model,
    "models/xgboost_model.pkl"
)

joblib.dump(
    list(X.columns),
    "models/feature_columns.pkl"
)

print("\nModel Saved Successfully!")
print("Saved to models/xgboost_model.pkl")
