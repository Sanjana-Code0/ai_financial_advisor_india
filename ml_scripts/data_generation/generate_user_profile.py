# ml_scripts/data_generation/generate_user_profile.py
import pandas as pd
import numpy as np
import os

# --- Configuration ---
NUM_USERS = 2500 # Keep this at 2500 or your desired number
DATA_DIR = 'data'
OUTPUT_FILE = os.path.join(DATA_DIR, 'user_profile_data_india.csv')
os.makedirs(DATA_DIR, exist_ok=True)

# Define categories
age_ranges = ['18-24', '25-34', '35-44', '45-54', '55+']
income_ranges = ['< ₹5 LPA', '₹5-12 LPA', '₹12-25 LPA', '₹25+ LPA']
savings_levels = ['Low', 'Medium', 'High']
debt_levels = ['Low', 'Medium', 'High']
dependents_options = ['Yes', 'No']
goal_options = ['Retirement', 'ChildEdu', 'Property', 'Marriage', 'Business', 'Wealth', 'Other']
time_horizon_map = {'< 5 years': 3, '5 - 10 years': 7, '11 - 15 years': 13, '16 - 20 years': 18, 'More than 20 years': 25}
time_horizon_choices = list(time_horizon_map.keys())
tolerance_levels = ['Low', 'Medium', 'High']
risk_profiles = ['Conservative', 'Moderate', 'Aggressive']

# --- NEW CATEGORIES ---
investment_knowledge_levels = ['Beginner', 'Intermediate', 'Advanced']
liquidity_needs_levels = ['Low', 'Medium', 'High'] # Low: can lock money, High: need access soon

# --- Generate Data ---
np.random.seed(42)
data = {
    'UserID': np.arange(1001, 1001 + NUM_USERS),
    'AgeRange': np.random.choice(age_ranges, NUM_USERS, p=[0.15, 0.25, 0.25, 0.20, 0.15]),
    'IncomeRange': np.random.choice(income_ranges, NUM_USERS, p=[0.3, 0.4, 0.2, 0.1]),
    'SavingsLevel': np.random.choice(savings_levels, NUM_USERS, p=[0.3, 0.5, 0.2]),
    'DebtLevel': np.random.choice(debt_levels, NUM_USERS, p=[0.4, 0.4, 0.2]),
    'HasDependents': np.random.choice(dependents_options, NUM_USERS, p=[0.6, 0.4]),
    'PrimaryGoal': np.random.choice(goal_options, NUM_USERS, p=[0.25, 0.2, 0.15, 0.1, 0.05, 0.2, 0.05]),
    'TimeHorizonRange': np.random.choice(time_horizon_choices, NUM_USERS, p=[0.1, 0.2, 0.2, 0.2, 0.3]),
    'SelfReportedTolerance': np.random.choice(tolerance_levels, NUM_USERS, p=[0.3, 0.5, 0.2]),
    # --- ADD NEW FIELDS ---
    'InvestmentKnowledge': np.random.choice(investment_knowledge_levels, NUM_USERS, p=[0.5, 0.3, 0.2]), # More beginners
    'LiquidityNeeds': np.random.choice(liquidity_needs_levels, NUM_USERS, p=[0.3, 0.4, 0.3]),
}
df = pd.DataFrame(data)
df['TimeHorizonYears'] = df['TimeHorizonRange'].map(time_horizon_map)
df = df.drop(columns=['TimeHorizonRange'])

# --- Generate Target Variable (RiskProfile) based on Rules ---
# IMPORTANT: Decide if these new fields should influence the main RiskProfile label.
# For now, let's assume the main RiskProfile is still based on the original factors,
# and the new fields are used to further refine investment choices *within* that risk profile.
# If you want them to affect the RiskProfile label, modify assign_risk_profile.
def assign_risk_profile(row): # Keep your existing well-tested rules for RiskProfile label
    if row['AgeRange'] == '55+' or row['TimeHorizonYears'] < 5 or \
       (row['DebtLevel'] == 'High' and row['SavingsLevel'] == 'Low'):
        return 'Conservative'
    if row['AgeRange'] in ['18-24', '25-34'] and \
       row['IncomeRange'] in ['₹12-25 LPA', '₹25+ LPA'] and \
       row['TimeHorizonYears'] > 15 and row['DebtLevel'] == 'Low':
        if row['SelfReportedTolerance'] == 'High': return 'Aggressive'
        elif row['SelfReportedTolerance'] == 'Medium': return np.random.choice(['Aggressive', 'Moderate'], p=[0.6, 0.4])
        else: return 'Moderate'
    if row['SelfReportedTolerance'] == 'High': return np.random.choice(['Moderate', 'Aggressive'], p=[0.6, 0.4])
    if row['SelfReportedTolerance'] == 'Low': return np.random.choice(['Conservative', 'Moderate'], p=[0.7, 0.3])
    return 'Moderate'
df['RiskProfile'] = df.apply(assign_risk_profile, axis=1)

# --- Save Data ---
df.to_csv(OUTPUT_FILE, index=False)
print(f"Generated {NUM_USERS} synthetic user profiles with new fields.")
print(f"Data saved to: {OUTPUT_FILE}")
print("\nColumns in generated data:", df.columns.tolist())
print(df.head())