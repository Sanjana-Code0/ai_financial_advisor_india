import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- Configuration ---
# Adjust paths relative to this script's location in ml_scripts/training/
DATA_DIR = 'data' # Input data directory relative to execution
MODELS_DIR = 'models' # Output models directory relative to execution
DATA_FILE = os.path.join(DATA_DIR, 'user_profile_data_india.csv') # Path to read input
PREPROCESSOR_FILE = os.path.join(MODELS_DIR, 'user_data_preprocessor.joblib') # Path to write output
MODEL_FILE = os.path.join(MODELS_DIR, 'risk_profile_rf_model.joblib') # Path to write output
# Make sure MODELS directory exists before writing
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Load Data ---
print(f"Loading data from {DATA_FILE}...")
try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_FILE}")
    print("Please run the data generation script first.")
    exit()
print("Loaded data shape:", df.shape)

# --- Define Features and Target ---
TARGET = 'RiskProfile'
# Define features explicitly based on the columns used to generate RiskProfile & available in input
# Exclude UserID and the target variable itself
features_to_use = [
    'AgeRange', 'IncomeRange', 'SavingsLevel', 'DebtLevel', 'HasDependents',
    'PrimaryGoal', 'TimeHorizonYears', 'SelfReportedTolerance'
]
X = df[features_to_use]
y = df[TARGET]

# Identify feature types FOR THE PREPROCESSOR based on columns in X
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=np.number).columns.tolist()

print(f"\nUsing Features: {features_to_use}")
print(f"Identified Categorical Features: {categorical_features}")
print(f"Identified Numerical Features: {numerical_features}")

# --- Create and Fit Preprocessing Pipeline ---
print("\nSetting up preprocessor...")
# Ensure the transformers list matches the identified types exactly
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='drop' # Explicitly drop any columns not specified (safer)
)

print("Fitting preprocessor...")
# Fit on the entire feature set X before splitting
preprocessor.fit(X)

# --- Save the Fitted Preprocessor ---
print(f"Saving preprocessor to {PREPROCESSOR_FILE}...")
joblib.dump(preprocessor, PREPROCESSOR_FILE)
print("Preprocessor saved.")

# --- Preprocess Data and Split ---
print("\nPreprocessing data...")
X_processed = preprocessor.transform(X)
print("Data preprocessed. Shape:", X_processed.shape)

print("\nSplitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# --- Train Random Forest Model ---
print("\nTraining Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)
print("Model training complete.")

# --- Evaluate Model ---
print("\nEvaluating model...")
y_pred_test = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_test, zero_division=0)) # Handle zero division

# --- Save the Trained Model ---
print(f"\nSaving trained model to {MODEL_FILE}...")
joblib.dump(rf_model, MODEL_FILE)
print("Model saved.")

print("\n--- Risk Model Training Script Finished ---")
