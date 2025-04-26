import pandas as pd
import numpy as np
import os

# --- Configuration ---
NUM_SAMPLES_PER_PROFILE = 1000 # How many investment examples per risk profile
DATA_DIR = 'data'
OUTPUT_FILE = os.path.join(DATA_DIR, 'investment_suitability_data_india.csv')
# Make sure directory exists before writing
os.makedirs(DATA_DIR, exist_ok=True)

# Define relevant categories (Match train_risk_model.py and user questions)
risk_profiles = ['Conservative', 'Moderate', 'Aggressive']

# Define example Indian Investment Types and their characteristics
investment_options = {
    'FD':           {'Volatility': 'Very Low', 'Return': 'Very Low'},
    'PPF':          {'Volatility': 'Very Low', 'Return': 'Low'}, # Often considered separate due to structure
    'DebtMF':       {'Volatility': 'Low',      'Return': 'Low'}, # Debt Mutual Fund
    'IndexFund':    {'Volatility': 'Medium',   'Return': 'Medium'}, # Nifty/Sensex ETF/Index Fund
    'BalancedMF':   {'Volatility': 'Medium',   'Return': 'Medium'}, # Balanced Advantage/Hybrid Fund
    'LargeCapMF':   {'Volatility': 'High',     'Return': 'High'}, # Large Cap Equity Fund
    'MidSmallCapMF':{'Volatility': 'Very High','Return': 'Very High'}, # Mid/Small Cap Equity Fund
    'DirectEquity': {'Volatility': 'Very High','Return': 'Very High'} # Direct Stocks
}
investment_types = list(investment_options.keys())
volatility_ranges = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
return_ranges = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

# --- Generate Data ---
np.random.seed(43) # Use a different seed
data_list = []

for profile in risk_profiles:
    for _ in range(NUM_SAMPLES_PER_PROFILE):
        inv_type = np.random.choice(investment_types)
        inv_details = investment_options[inv_type]

        # Rule-based Suitability (TARGET)
        suitability = 'Not Suitable' # Default
        volatility = inv_details['Volatility']

        if profile == 'Conservative' and volatility in ['Very Low', 'Low']:
            suitability = 'Suitable'
        elif profile == 'Moderate' and volatility in ['Low', 'Medium', 'High']: # Moderate can take some high vol
             suitability = 'Suitable'
        elif profile == 'Aggressive' and volatility in ['Medium', 'High', 'Very High']:
            suitability = 'Suitable'

        # Add some noise/randomness maybe? Optional.

        data_list.append({
            'RiskProfile': profile,
            'InvestmentType': inv_type,
            'InvestmentVolRange': inv_details['Volatility'], # Feature
            'InvestmentRetRange': inv_details['Return'],   # Feature (less used in rule)
            'Suitability': suitability # Target
        })

df = pd.DataFrame(data_list)

# --- Save Data ---
df.to_csv(OUTPUT_FILE, index=False)

print(f"Generated {len(df)} investment suitability samples.")
print(f"Data saved to: {OUTPUT_FILE}")
print("\nSample Data:")
print(df.head())
print("\nSuitability Distribution:")
print(df['Suitability'].value_counts(normalize=True))