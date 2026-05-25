# ==========================================
# Feature Engineering for Predictive Maintenance
# ==========================================

import pandas as pd
import numpy as np

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("data/PdM_telemetry.csv")

print("Original Shape:", df.shape)

# ==========================================
# Convert Datetime
# ==========================================

df['datetime'] = pd.to_datetime(df['datetime'])

# Sort data
df = df.sort_values(['machineID', 'datetime'])

# ==========================================
# Time Features
# ==========================================

df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day
df['month'] = df['datetime'].dt.month
df['day_of_week'] = df['datetime'].dt.dayofweek

# ==========================================
# Lag Features
# ==========================================

sensor_cols = [
    'volt',
    'rotate',
    'pressure',
    'vibration'
]

for col in sensor_cols:

    df[f'{col}_lag_1'] = (
        df.groupby('machineID')[col]
        .shift(1)
    )

    df[f'{col}_lag_2'] = (
        df.groupby('machineID')[col]
        .shift(2)
    )

# ==========================================
# Rolling Window Features
# ==========================================

windows = [3, 6, 12]

for col in sensor_cols:

    for window in windows:

        # Rolling Mean
        df[f'{col}_rolling_mean_{window}'] = (
            df.groupby('machineID')[col]
            .transform(
                lambda x: x.rolling(window).mean()
            )
        )

        # Rolling Standard Deviation
        df[f'{col}_rolling_std_{window}'] = (
            df.groupby('machineID')[col]
            .transform(
                lambda x: x.rolling(window).std()
            )
        )

# ==========================================
# Exponential Moving Average
# ==========================================

for col in sensor_cols:

    df[f'{col}_ema_5'] = (
        df.groupby('machineID')[col]
        .transform(
            lambda x: x.ewm(span=5, adjust=False).mean()
        )
    )

# ==========================================
# Interaction Features
# ==========================================

df['power'] = df['volt'] * df['rotate']

df['stress'] = (
    df['pressure'] * df['vibration']
)

# ==========================================
# Maintenance History Features
# ==========================================

# Load maintenance dataset
maint = pd.read_csv("data/PdM_maint.csv")

maint['datetime'] = pd.to_datetime(
    maint['datetime']
)

# Count maintenance events
maint_count = (
    maint.groupby('machineID')
    .size()
    .reset_index(name='maintenance_count')
)

# Merge with telemetry
df = df.merge(
    maint_count,
    on='machineID',
    how='left'
)

# Fill missing values
df['maintenance_count'] = (
    df['maintenance_count']
    .fillna(0)
)

# ==========================================
# Failure Features
# ==========================================

failures = pd.read_csv("data/PdM_failures.csv")

failure_count = (
    failures.groupby('machineID')
    .size()
    .reset_index(name='failure_count')
)

df = df.merge(
    failure_count,
    on='machineID',
    how='left'
)

df['failure_count'] = (
    df['failure_count']
    .fillna(0)
)

# ==========================================
# Handle Missing Values
# ==========================================

df.fillna(method='bfill', inplace=True)
df.fillna(method='ffill', inplace=True)
df.fillna(0, inplace=True)

print("Final Shape:", df.shape)


# ==========================================
# Save Final Dataset
# ==========================================

df.to_csv(
    "data/final_train.csv",
    index=False
)

print("Feature engineering completed successfully!")
