# Predictive Maintenance

## Project Overview

Azure Predictive Maintenance is a machine learning project designed to predict machine failures before they occur by analyzing telemetry and maintenance data. The system leverages advanced feature engineering, class imbalance handling, hyperparameter tuning, and model explainability techniques to improve maintenance planning and reduce operational downtime.

## Features

* Data Preprocessing and Cleaning
* Advanced Feature Engineering

  * Lag Features
  * Rolling Statistics
  * Exponential Moving Averages (EMA)
  * Interaction Features
* Imbalance Handling using SMOTE
* XGBoost Model Training
* Hyperparameter Tuning using GridSearchCV
* Model Explainability using SHAP
* Interactive Streamlit Dashboard
* Model Serialization using Joblib

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Imbalanced-learn (SMOTE)
* SHAP
* Matplotlib
* Streamlit
* Joblib

## Project Structure

```text
AzurePredictiveMaintenance/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── final_train.csv
│   └── raw_data.csv
│
├── models/
│   ├── xgboost_model.pkl
│   └── feature_columns.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   └── shap_analysis.py
│
├── requirements.txt
└── README.md
```

## Evaluation Metrics

The model is evaluated using:

* Precision
* Recall
* F1-Score
* PR-AUC
* Confusion Matrix

## Future Enhancements

* Real-time IoT Data Integration
* Azure Cloud Deployment
* Docker Containerization
* CI/CD Pipeline Integration
* Deep Learning-based Failure Prediction

## Results

The model successfully identifies potential machine failures using sensor telemetry data and historical maintenance information, enabling proactive maintenance and reducing unexpected equipment downtime.
