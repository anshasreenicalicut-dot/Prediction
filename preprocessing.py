import pandas as pd
import os

# ==========================================
# Load Datasets
# ==========================================

telemetry = pd.read_csv("data/PdM_telemetry.csv")
failures = pd.read_csv("data/PdM_failures.csv")
machines = pd.read_csv("data/PdM_machines.csv")
maintenance = pd.read_csv("data/PdM_maint.csv")
errors = pd.read_csv("data/PdM_errors.csv")

print("Datasets Loaded Successfully!")

# ==========================================
# Convert Datetime Columns
# ==========================================

telemetry["datetime"] = pd.to_datetime(
    telemetry["datetime"]
)

failures["datetime"] = pd.to_datetime(
    failures["datetime"]
)

maintenance["datetime"] = pd.to_datetime(
    maintenance["datetime"]
)

errors["datetime"] = pd.to_datetime(
    errors["datetime"]
)

# ==========================================
# Merge Machine Information
# ==========================================

df = telemetry.merge(
    machines,
    on="machineID",
    how="left"
)

# ==========================================
# Failure Count
# ==========================================

failure_count = (
    failures.groupby(
        ["machineID", "datetime"]
    )
    .size()
    .reset_index(name="failure_count")
)

df = df.merge(
    failure_count,
    on=["machineID", "datetime"],
    how="left"
)

# ==========================================
# Maintenance Count
# ==========================================

maintenance_count = (
    maintenance.groupby(
        ["machineID", "datetime"]
    )
    .size()
    .reset_index(name="maintenance_count")
)

df = df.merge(
    maintenance_count,
    on=["machineID", "datetime"],
    how="left"
)

# ==========================================
# Error Count
# ==========================================

error_count = (
    errors.groupby(
        ["machineID", "datetime"]
    )
    .size()
    .reset_index(name="error_count")
)

df = df.merge(
    error_count,
    on=["machineID", "datetime"],
    how="left"
)

# ==========================================
# Handle Missing Values
# ==========================================

df["failure_count"] = df[
    "failure_count"
].fillna(0)

df["maintenance_count"] = df[
    "maintenance_count"
].fillna(0)

df["error_count"] = df[
    "error_count"
].fillna(0)

# ==========================================
# Encode Model Type
# ==========================================

if "model" in df.columns:
    df["model"] = (
        df["model"]
        .astype("category")
        .cat.codes
    )

# ==========================================
# Save Cleaned Dataset
# ==========================================

os.makedirs(
    "data",
    exist_ok=True
)

df.to_csv(
    "data/cleaned_train.csv",
    index=False
)

print("\nPreprocessing Completed!")
print("Saved: data/cleaned_train.csv")
print("Final Shape:", df.shape)