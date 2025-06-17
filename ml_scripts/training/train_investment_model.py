# ml_scripts/training/train_investment_model.py
import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report

# --- Configuration ---
# Paths are relative to the project root (ai_financial_advisor_india)
# where you execute 'python ml_scripts/training/train_investment_model.py'
DATA_DIR = 'data'
MODELS_DIR = 'models'
DATA_FILE = os.path.join(DATA_DIR, 'investment_suitability_data_india.csv') # Path to READ investment data
PREPROCESSOR_FILE = os.path.join(MODELS_DIR, 'investment_data_preprocessor.joblib') # Path to WRITE
MODEL_FILE = os.path.join(MODELS_DIR, 'investment_suitability_xgb_model.joblib') # Path to WRITE
os.makedirs(MODELS_DIR, exist_ok=True) # Ensure models directory exists

# --- Load Data ---
print(f"Loading data from {DATA_FILE}...")
try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"ERROR: Data file not found at '{DATA_FILE}'.")
    print("Please ensure 'generate_investment_data.py' has run successfully and created this file in the 'data' directory of your project root.")
    exit()
print("Loaded data shape:", df.shape)

# --- Define Features and Target ---
TARGET = 'Suitability'
features_to_use = [
    'RiskProfile', 'InvestmentKnowledge', 'LiquidityNeeds', 'TimeHorizonYears',
    'InvestmentType', 'InvestmentVolRange', 'InvestmentRetRange'
]
X = df[features_to_use]
y = df[TARGET]

categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=np.number).columns.tolist()

print(f"\nUsing Features: {features_to_use}")
print(f"Categorical Features for Investment Model: {categorical_features}")
print(f"Numerical Features for Investment Model: {numerical_features}")

# --- Create and Fit Preprocessing Pipeline ---
print("\nSetting up preprocessor for investment model...")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='drop'
)
print("Fitting investment preprocessor...")
preprocessor.fit(X)
print(f"Fitted investment preprocessor features: {preprocessor.get_feature_names_out()}")

# --- Save the Fitted Preprocessor ---
print(f"Saving investment preprocessor to {PREPROCESSOR_FILE}...")
joblib.dump(preprocessor, PREPROCESSOR_FILE)
print("Preprocessor saved.")

# --- Preprocess Data and Split ---
print("\nPreprocessing data...")
X_processed = preprocessor.transform(X)
y_mapped = y.map({'Suitable': 1, 'Not Suitable': 0})
print("Data preprocessed. Input Shape:", X_processed.shape, "Target Shape:", y_mapped.shape)

print("\nSplitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_mapped, test_size=0.2, random_state=43, stratify=y_mapped)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# --- Train XGBoost Model ---
print("\nTraining XGBoost model...")
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=43)
xgb_model.fit(X_train, y_train)
print("Model training complete.")

# --- Evaluate Model ---
print("\nEvaluating investment model...")
y_pred_test = xgb_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
print("\nClassification Report (Test Set - Mapped Target 0/1):")
print(classification_report(y_test, y_pred_test, zero_division=0))

# --- Save the Trained Model ---
print(f"Saving trained XGBoost model to {MODEL_FILE}...")
joblib.dump(xgb_model, MODEL_FILE)
print("Model saved.")
print("\n--- Investment Model Training Script Finished ---")