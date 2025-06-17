# ml_scripts/data_generation/generate_investment_data.py
import pandas as pd
import numpy as np
import os

# --- Configuration ---
# Paths are relative to the project root (ai_financial_advisor_india)
# where you execute 'python ml_scripts/data_generation/generate_investment_data.py'
USER_PROFILE_DATA_FILE = os.path.join('data', 'user_profile_data_india.csv') # Path to READ user profiles
DATA_DIR = 'data'                                                          # Directory to WRITE output
OUTPUT_FILE = os.path.join(DATA_DIR, 'investment_suitability_data_india.csv') # Path to WRITE investment data
os.makedirs(DATA_DIR, exist_ok=True) # Ensure data directory exists

# Define investment options (same as in prediction.py)
AVAILABLE_INVESTMENTS = {
    'FD':           {'Volatility': 'Very Low', 'Return': 'Very Low'},
    'PPF':          {'Volatility': 'Very Low', 'Return': 'Low'},
    'DebtMF':       {'Volatility': 'Low',      'Return': 'Low'},
    'IndexFund':    {'Volatility': 'Medium',   'Return': 'Medium'},
    'BalancedMF':   {'Volatility': 'Medium',   'Return': 'Medium'},
    'LargeCapMF':   {'Volatility': 'High',     'Return': 'High'},
    'MidSmallCapMF':{'Volatility': 'Very High','Return': 'Very High'},
    'DirectEquity': {'Volatility': 'Very High','Return': 'Very High'}
}
investment_types = list(AVAILABLE_INVESTMENTS.keys())

# --- Load User Profile Data ---
print(f"Attempting to load user profiles from: {USER_PROFILE_DATA_FILE}")
try:
    user_profiles_df = pd.read_csv(USER_PROFILE_DATA_FILE)
except FileNotFoundError:
    print(f"ERROR: User profile data not found at '{USER_PROFILE_DATA_FILE}'.")
    print("Please ensure 'generate_user_profile.py' has run successfully and created this file in the 'data' directory of your project root.")
    exit()
print(f"Successfully loaded {len(user_profiles_df)} user profiles.")

# --- Generate Investment Suitability Data ---
np.random.seed(44)
investment_data_list = []

for index, user_row in user_profiles_df.iterrows():
    user_risk_profile = user_row['RiskProfile']
    user_knowledge = user_row['InvestmentKnowledge']
    user_liquidity = user_row['LiquidityNeeds']
    user_time_horizon = user_row['TimeHorizonYears']

    for inv_type in investment_types:
        inv_details = AVAILABLE_INVESTMENTS[inv_type]
        inv_volatility = inv_details['Volatility']
        suitability = 'Not Suitable' # Default

        # --- More NUANCED Suitability Rules (TARGET) ---
        if user_risk_profile == 'Conservative':
            if inv_volatility in ['Very Low', 'Low']: suitability = 'Suitable'
            if inv_type == 'PPF' and (user_liquidity == 'High' or user_time_horizon < 5): suitability = 'Not Suitable'
        elif user_risk_profile == 'Moderate':
            if inv_volatility in ['Low', 'Medium']: suitability = 'Suitable'
            elif inv_volatility == 'High' and user_knowledge in ['Intermediate', 'Advanced'] and user_time_horizon > 7: suitability = 'Suitable'
            if inv_volatility == 'Very High' and user_knowledge == 'Advanced' and user_liquidity == 'Low' and user_time_horizon > 10: suitability = 'Suitable'
        elif user_risk_profile == 'Aggressive':
            if inv_volatility in ['Medium', 'High']: suitability = 'Suitable'
            elif inv_volatility == 'Very High':
                if user_knowledge in ['Intermediate', 'Advanced'] and user_liquidity != 'High' and user_time_horizon > 5: suitability = 'Suitable'
                elif user_knowledge == 'Beginner' and user_time_horizon > 10: suitability = 'Suitable'
            if inv_type == 'DirectEquity' and user_knowledge == 'Beginner' and user_time_horizon < 7: suitability = 'Not Suitable'

        investment_data_list.append({
            'RiskProfile': user_risk_profile, 'InvestmentKnowledge': user_knowledge,
            'LiquidityNeeds': user_liquidity, 'TimeHorizonYears': user_time_horizon,
            'InvestmentType': inv_type, 'InvestmentVolRange': inv_volatility,
            'InvestmentRetRange': inv_details['Return'], 'Suitability': suitability
        })

df_investment_suitability = pd.DataFrame(investment_data_list)

# --- Save Data ---
df_investment_suitability.to_csv(OUTPUT_FILE, index=False)
print(f"Generated {len(df_investment_suitability)} investment suitability samples.")
print(f"Data saved to: {OUTPUT_FILE}")
print("\nColumns in generated investment data:", df_investment_suitability.columns.tolist())
print(df_investment_suitability.head())
print("\nSuitability Distribution:")
print(df_investment_suitability['Suitability'].value_counts(normalize=True))