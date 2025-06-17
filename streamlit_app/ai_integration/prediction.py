# streamlit_app/ai_integration/prediction.py
# (Keep all existing imports and other functions like load_ai_components, format_shap, get_risk_profile)
# ... (Imports and Config, RISK_FEATURE_ORDER, AVAILABLE_INVESTMENTS, load_ai_components, format_shap_explanation_user_focused, get_risk_profile_and_explanation) ...
import joblib, pandas as pd, numpy as np, shap, os, streamlit as st, xgboost as xgb, traceback
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); MODEL_DIR = os.path.join(PROJECT_ROOT_DIR, 'models')
RISK_PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'user_data_preprocessor.joblib'); RISK_MODEL_PATH = os.path.join(MODEL_DIR, 'risk_profile_rf_model.joblib')
INV_PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'investment_data_preprocessor.joblib'); INV_MODEL_PATH = os.path.join(MODEL_DIR, 'investment_suitability_xgb_model.joblib')
RISK_FEATURE_ORDER = ['AgeRange', 'IncomeRange', 'SavingsLevel', 'DebtLevel', 'HasDependents','PrimaryGoal', 'TimeHorizonYears', 'SelfReportedTolerance']
# --- *** UPDATED INVESTMENT FEATURE ORDER *** ---
INV_FEATURE_ORDER = [
    'RiskProfile',
    'InvestmentKnowledge',
    'LiquidityNeeds',
    'TimeHorizonYears', # Numerical, will be scaled by preprocessor
    'InvestmentType',
    'InvestmentVolRange',
    'InvestmentRetRange'
]
# --- *** END UPDATE *** ---
AVAILABLE_INVESTMENTS = {'FD':{'Volatility':'Very Low','Return':'Very Low'},'PPF':{'Volatility':'Very Low','Return':'Low'},'DebtMF':{'Volatility':'Low','Return':'Low'},'IndexFund':{'Volatility':'Medium','Return':'Medium'},'BalancedMF':{'Volatility':'Medium','Return':'Medium'},'LargeCapMF':{'Volatility':'High','Return':'High'},'MidSmallCapMF':{'Volatility':'Very High','Return':'Very High'},'DirectEquity':{'Volatility':'Very High','Return':'Very High'}}
@st.cache_resource
def load_ai_components():
    # ...(Same robust loading logic as before)...
    print("Attempting.. AI components.."); components = {"risk_preprocessor":None,"risk_model":None,"risk_explainer":None,"inv_preprocessor":None,"inv_model":None,"inv_explainer":None,"risk_feature_names":None,"inv_feature_names":None,"load_error":None}
    try:
        if os.path.exists(RISK_PREPROCESSOR_PATH): components["risk_preprocessor"]=joblib.load(RISK_PREPROCESSOR_PATH);print("-> Risk preproc loaded.");_try_get_feature_names(components, "risk_preprocessor", "risk_feature_names")
        else: raise FileNotFoundError(f"Risk preproc missing: {RISK_PREPROCESSOR_PATH}")
        if os.path.exists(RISK_MODEL_PATH): components["risk_model"]=joblib.load(RISK_MODEL_PATH); print("-> Risk model loaded."); _try_init_explainer(components, "risk_model", "risk_explainer")
        else: raise FileNotFoundError(f"Risk model missing: {RISK_MODEL_PATH}")
        if os.path.exists(INV_PREPROCESSOR_PATH): components["inv_preprocessor"]=joblib.load(INV_PREPROCESSOR_PATH); print("-> Inv preproc loaded."); _try_get_feature_names(components, "inv_preprocessor", "inv_feature_names")
        else: raise FileNotFoundError(f"Inv preproc missing: {INV_PREPROCESSOR_PATH}")
        if os.path.exists(INV_MODEL_PATH): components["inv_model"]=joblib.load(INV_MODEL_PATH); print("-> Inv model loaded."); _try_init_explainer(components, "inv_model", "inv_explainer")
        else: raise FileNotFoundError(f"Inv model missing: {INV_MODEL_PATH}")
        print("--- AI loading OK ---")
    except Exception as e: critical_error=f"AI Loading Error: {e}"; print(f"!!! {critical_error} !!!"); components["load_error"]=critical_error; st.error(critical_error)
    return components
def _try_get_feature_names(components, preprocessor_key, feature_names_key):
    try: components[feature_names_key] = components[preprocessor_key].get_feature_names_out(); print(f"-> {feature_names_key.replace('_',' ')} ({len(components[feature_names_key])})")
    except Exception as e: print(f"Warning: Could not get {feature_names_key}: {e}")
def _try_init_explainer(components, model_key, explainer_key):
    if components[model_key]:
        try: components[explainer_key] = shap.TreeExplainer(components[model_key]); print(f"-> SHAP {model_key.replace('_',' ')} explainer init.")
        except Exception as e: print(f"Error initializing SHAP {model_key.replace('_',' ')} explainer: {e}")
AI_COMPONENTS = load_ai_components()
# ...(Keep format_shap_explanation_user_focused as before)...
def format_shap_explanation_user_focused(shap_values_instance, preprocessor_feature_names, original_input_dict, predicted_outcome_label, explanation_type='risk', max_features=3):
    if explanation_type == 'risk': intro = f"Here's what primarily led to the **'{predicted_outcome_label}'** risk profile assessment:\n\n"
    elif explanation_type == 'investment': intro = f"Here's why this investment is considered **'{predicted_outcome_label}'** for you:\n\n"
    else: intro = "**Key factors in this decision:**\n\n"
    no_detail_msg = "*We considered your overall profile, but a detailed breakdown is unavailable.*"; print(f"--- format_shap_explanation_user_focused for {explanation_type} ---")
    try:
        if preprocessor_feature_names is None or shap_values_instance is None or len(preprocessor_feature_names)==0 or len(shap_values_instance)==0 or original_input_dict is None: print("DEBUG: format_user_focused - Missing inputs."); return no_detail_msg
        if len(preprocessor_feature_names) != len(shap_values_instance): print(f"ERROR: format_user_focused - Mismatch length."); return "Explanation error."
        contributions = {}
        for i, full_feature_name in enumerate(preprocessor_feature_names):
            parts = full_feature_name.split('__');
            if len(parts) < 2:
                if full_feature_name in original_input_dict: contributions[full_feature_name] = contributions.get(full_feature_name, 0) + shap_values_instance[i]
                continue
            transformer_type = parts[0]; original_feature_name_with_suffix = parts[1]
            split_suffix = original_feature_name_with_suffix.rsplit('_', 1); base_feature_name = split_suffix[0]; encoded_value_part = split_suffix[1] if len(split_suffix) > 1 else original_feature_name_with_suffix
            if transformer_type == 'cat':
                if base_feature_name in original_input_dict and str(original_input_dict[base_feature_name]) == encoded_value_part: contributions[base_feature_name] = contributions.get(base_feature_name, 0) + shap_values_instance[i]
            elif transformer_type == 'num':
                if base_feature_name in original_input_dict: contributions[base_feature_name] = contributions.get(base_feature_name, 0) + shap_values_instance[i]
        if not contributions: print("DEBUG: format_user_focused - No original features mapped."); return no_detail_msg
        df_contributions = pd.DataFrame(list(contributions.items()), columns=['original_feature', 'total_shap_value']); df_contributions['abs_shap'] = np.abs(df_contributions['total_shap_value'])
        df_contributions = df_contributions.sort_values(by='abs_shap', ascending=False).reset_index(drop=True); df_contributions = df_contributions[df_contributions['abs_shap'] > 0.01]
        if df_contributions.empty: print("DEBUG: format_user_focused - No significant factors after filtering."); return "*Overall profile considered, no single input stood out significantly.*"
        explanation_points = []
        for i, row in enumerate(df_contributions.head(max_features).itertuples()):
            original_feature = row.original_feature; user_value = original_input_dict.get(original_feature); shap_value = row.total_shap_value
            friendly_name_map = {'AgeRange':'Your age group','IncomeRange':'Your income level','SavingsLevel':'Your savings level','DebtLevel':'Your debt level','HasDependents':'Having dependents','PrimaryGoal':'Your primary goal','TimeHorizonYears':'Your investment time horizon','SelfReportedTolerance':'Your stated risk comfort','RiskProfile':"Your overall risk profile",'InvestmentType':"The type of investment",'InvestmentVolRange':"Investment's typical volatility",'InvestmentRetRange':"Investment's potential return"}
            friendly_name = friendly_name_map.get(original_feature, original_feature.replace('_', ' ').capitalize()); value_display = f" ('{user_value}')" if user_value is not None else ""
            if original_feature == 'HasDependents': value_display = f" ({'Yes' if user_value == 'Yes' else 'No'})"
            point = ""
            if explanation_type == 'risk':
                if shap_value > 0.02: point = f"*   **{friendly_name}{value_display}:** Key factor for '{predicted_outcome_label}' profile."
                elif shap_value > 0.01: point = f"*   **{friendly_name}{value_display}:** Aligns with '{predicted_outcome_label}' approach."
            elif explanation_type == 'investment':
                if shap_value > 0.02: point = f"*   **{friendly_name}{value_display}:** Strong reason this investment is suitable."
                elif shap_value > 0.01: point = f"*   **{friendly_name}{value_display}:** Supports this investment's suitability."
            if point: explanation_points.append(point)
        if not explanation_points: return "*Profile assessed, factors had mixed influence.*"
        return intro + "\n".join(explanation_points)
    except Exception as e: print(f"ERROR in format_shap: {e}"); traceback.print_exc(); return no_detail_msg

# --- Risk Profile Prediction Function (Calls user-focused formatter) ---
def get_risk_profile_and_explanation(user_profile_dict):
    # ...(Function logic remains the same as the previous working version)...
    print("--- Running Risk Prediction (with user-focused SHAP) ---")
    preprocessor = AI_COMPONENTS.get("risk_preprocessor"); model = AI_COMPONENTS.get("risk_model"); explainer = AI_COMPONENTS.get("risk_explainer"); preprocessor_feature_names = AI_COMPONENTS.get("risk_feature_names"); load_error = AI_COMPONENTS.get("load_error") # Changed feature_names to preprocessor_feature_names
    if load_error or not all([preprocessor, model]): return None
    print(f"Risk Pred: Received profile keys: {list(user_profile_dict.keys())}")
    try:
        missing_keys = set(RISK_FEATURE_ORDER) - set(user_profile_dict.keys())
        if missing_keys: print(f"Error: Missing keys {missing_keys}"); st.error(f"Missing info: {missing_keys}"); return None
        input_df = pd.DataFrame([user_profile_dict], columns=RISK_FEATURE_ORDER)
        processed_input = preprocessor.transform(input_df)
        classes = model.classes_ ; prediction_label = model.predict(processed_input)[0]
        try: predicted_class_index = np.where(classes == prediction_label)[0][0]
        except IndexError: print(f"Error: Label '{prediction_label}' not in classes '{classes}'"); st.error("Prediction error."); return None
        print(f"Risk Pred: Raw prediction: {prediction_label} (Index: {predicted_class_index})")
        explanation_text = "*Detailed factor analysis unavailable.*"
        if explainer and preprocessor_feature_names is not None:
            try:
                print(f"Risk Pred: Calculating SHAP values...")
                shap_values = explainer.shap_values(processed_input)
                shap_values_instance = None
                if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                     if 0 <= predicted_class_index < shap_values.shape[2]: shap_values_instance = shap_values[0, :, predicted_class_index]
                elif isinstance(shap_values, list) and len(shap_values) == len(classes):
                     if 0 <= predicted_class_index < len(shap_values): shap_values_instance = shap_values[predicted_class_index][0]
                else: print(f"Warning: Unexpected SHAP format for risk.")
                if shap_values_instance is not None:
                     print(f"Risk Pred: Extracted SHAP values. Shape: {shap_values_instance.shape}.")
                     explanation_text = format_shap_explanation_user_focused(shap_values_instance, preprocessor_feature_names, user_profile_dict, prediction_label, explanation_type='risk')
                else: explanation_text = "*Could not process risk explanation format.*"
            except Exception as shap_e: print(f"Risk Pred: SHAP calculation failed: {shap_e}"); traceback.print_exc(); explanation_text = "*Error generating risk factors.*"
        elif not explainer: explanation_text = "*Explanation unavailable (explainer).*"
        else: explanation_text = "*Explanation unavailable (feature names).*"
        print(f"--- Risk Prediction Finished: {prediction_label} ---")
        return {'prediction': str(prediction_label), 'explanation': explanation_text}
    except Exception as e: error_msg = f"Error during risk prediction: {e}"; print(error_msg); traceback.print_exc(); st.error(error_msg); return None


# --- *** MODIFIED: Investment Recommendation Function *** ---
def get_investment_recommendations_and_explanation(user_profile_dict_full, user_risk_profile: str): # Takes full user profile
    """
    Predicts suitability for available investments based on user's FULL profile
    and generates USER-FOCUSED explanations for suitable ones.
    """
    print(f"\n--- Running Investment Recommendations for Profile: {user_risk_profile} ---")
    preprocessor = AI_COMPONENTS.get("inv_preprocessor")
    model = AI_COMPONENTS.get("inv_model")
    explainer = AI_COMPONENTS.get("inv_explainer")
    preprocessor_feature_names = AI_COMPONENTS.get("inv_feature_names") # Use names from preprocessor
    load_error = AI_COMPONENTS.get("load_error")

    if load_error or not all([preprocessor, model]):
        return [{"investment": "Error", "explanation": "AI components missing."}]

    recommendations = []
    for inv_type, details in AVAILABLE_INVESTMENTS.items():
        try:
            # --- Construct input_data with ALL relevant fields for the investment model ---
            input_data = {
                'RiskProfile': user_risk_profile, # User's overall risk profile
                'InvestmentKnowledge': user_profile_dict_full.get('InvestmentKnowledge'),
                'LiquidityNeeds': user_profile_dict_full.get('LiquidityNeeds'),
                'TimeHorizonYears': user_profile_dict_full.get('TimeHorizonYears'), # User's actual time horizon
                'InvestmentType': inv_type,                 # Current investment being evaluated
                'InvestmentVolRange': details['Volatility'],  # Characteristic of inv_type
                'InvestmentRetRange': details['Return']       # Characteristic of inv_type
            }
            # Ensure all keys in INV_FEATURE_ORDER are present in input_data, even if None
            for key in INV_FEATURE_ORDER:
                input_data.setdefault(key, None)


            input_df = pd.DataFrame([input_data], columns=INV_FEATURE_ORDER) # Use updated INV_FEATURE_ORDER
            # print(f"Inv Rec: Checking {inv_type}. Input for model:\n{input_df}") # Debug

            processed_input = preprocessor.transform(input_df)
            prediction_code = model.predict(processed_input)[0]
            suitability = 'Suitable' if prediction_code == 1 else 'Not Suitable'

            if suitability == 'Suitable':
                explanation_text = "*Detailed rationale unavailable.*"
                if explainer and preprocessor_feature_names is not None:
                    try:
                        print(f"Inv Rec: Calculating SHAP for {inv_type}...")
                        shap_values = explainer.shap_values(processed_input)
                        shap_values_instance = None
                        if isinstance(shap_values, np.ndarray) and shap_values.ndim >= 1:
                            shap_values_instance = shap_values[0]
                        else: print(f"Warning: Unexpected SHAP format for {inv_type}.")

                        if shap_values_instance is not None:
                             print(f"Inv Rec: Extracted SHAP values for {inv_type}. Shape: {shap_values_instance.shape}.")
                             # Call USER-FOCUSED formatter, pass the 'input_data' for this investment
                             explanation_text = format_shap_explanation_user_focused(
                                 shap_values_instance, preprocessor_feature_names, input_data, "Suitable", explanation_type='investment'
                             )
                        else: explanation_text = "*Could not process investment explanation format.*"
                    except Exception as shap_e: print(f"Inv Rec: SHAP failed for {inv_type}: {shap_e}"); traceback.print_exc(); explanation_text = "*Error generating rationale.*"
                elif not explainer: explanation_text = "*Rationale unavailable (explainer).*"
                else: explanation_text = "*Rationale unavailable (feature names).*"

                recommendations.append({"investment": inv_type, "suitability": suitability, "explanation": explanation_text})
        except Exception as e: error_msg = f"Error predicting suitability for {inv_type}: {e}"; print(error_msg); traceback.print_exc()

    print(f"--- Finished Generating Investment Recs. Found {len(recommendations)} suitable. ---")
    if not recommendations: return [{"investment": "None Suitable", "explanation": "*Based on the analysis, no standard investments were deemed suitable.*"}]
    return recommendations
# --- *** End of Modified Investment Function *** ---

# --- Placeholder Function for RL Planning (Keep as is) ---
def get_planning_recommendation(user_profile_dict, risk_profile, suitable_investments):
    # ...(placeholder logic remains the same)...
    print(f"--- Generating Planning Recommendation (Placeholder) ---")
    plan_actions = [f"Suggested Monthly Savings: 15% of income (Placeholder)", f"Investment Allocation: 60% IndexFund, 40% LargeCapMF (Placeholder)"]
    plan_explanation = f"This placeholder plan balances growth for your '{user_profile_dict.get('PrimaryGoal')}' goal with your '{risk_profile}' profile. (Reinforcement Learning agent integration pending)."
    print("--- Planning Recommendation Placeholder Generated ---")
    return {"actions": plan_actions, "explanation": plan_explanation}

# At the top of prediction.py or in a separate config.py
# These are ILLUSTRATIVE. Research appropriate long-term averages for Indian markets.
INVESTMENT_RETURN_MAPPING = {
    "Very Low": 0.04,  # 4%
    "Low": 0.06,       # 6%
    "Medium": 0.08,    # 8%
    "High": 0.12,      # 12%
    "Very High": 0.15  # 15%
}

# streamlit_app/ai_integration/prediction.py
# ... (Keep all existing imports, configurations, and functions) ...

# (Make sure INVESTMENT_RETURN_MAPPING is defined above this function)

def project_investment_growth(principal_amount, annual_return_rate, years, compounding_frequency='annually'):
    """
    Calculates projected investment growth.

    Args:
        principal_amount (float): The initial amount invested.
        annual_return_rate (float): The average annual rate of return (e.g., 0.08 for 8%).
        years (int): The number of years to project.
        compounding_frequency (str): 'annually', 'semi-annually', 'quarterly', 'monthly'.

    Returns:
        tuple: (projected_value, total_growth)
    """
    if annual_return_rate is None: # Handle cases where mapping might fail
        return principal_amount, 0

    if compounding_frequency == 'annually':
        n = 1
    elif compounding_frequency == 'semi-annually':
        n = 2
    elif compounding_frequency == 'quarterly':
        n = 4
    elif compounding_frequency == 'monthly':
        n = 12
    else: # Default to annually if invalid
        n = 1

    # Compound interest formula: A = P(1 + r/n)^(nt)
    try:
        projected_value = principal_amount * (1 + (annual_return_rate / n))**(n * years)
        total_growth = projected_value - principal_amount
        return round(projected_value, 2), round(total_growth, 2)
    except Exception as e:
        print(f"Error in projection calculation: {e}")
        return principal_amount, 0 # Return principal if calculation fails

# --- Modify get_investment_recommendations_and_explanation ---
# It will now add projection data to each suitable investment
def get_investment_recommendations_and_explanation(user_profile_dict_full, user_risk_profile: str,
                                                   projection_principal=100000, projection_years=5): # Add default projection params
    """
    Predicts suitability, generates explanations, AND ADDS PROJECTED GROWTH.
    """
    print(f"\n--- Running Investment Recommendations for Profile: {user_risk_profile} ---")
    # ... (Keep existing component loading and error checks) ...
    preprocessor=AI_COMPONENTS.get("inv_preprocessor"); model=AI_COMPONENTS.get("inv_model")
    explainer=AI_COMPONENTS.get("inv_explainer"); preprocessor_feature_names=AI_COMPONENTS.get("inv_feature_names")
    load_error=AI_COMPONENTS.get("load_error")
    if load_error or not all([preprocessor,model]): return [{"investment": "Error", "explanation": "AI components missing."}]

    recommendations = []
    for inv_type, details in AVAILABLE_INVESTMENTS.items():
        try:
            input_data = {
                'RiskProfile': user_risk_profile,
                'InvestmentKnowledge': user_profile_dict_full.get('InvestmentKnowledge'),
                'LiquidityNeeds': user_profile_dict_full.get('LiquidityNeeds'),
                'TimeHorizonYears': user_profile_dict_full.get('TimeHorizonYears'),
                'InvestmentType': inv_type,
                'InvestmentVolRange': details['Volatility'],
                'InvestmentRetRange': details['Return'] # This is the key for projection
            }
            for key in INV_FEATURE_ORDER: input_data.setdefault(key, None)
            input_df = pd.DataFrame([input_data], columns=INV_FEATURE_ORDER)
            processed_input = preprocessor.transform(input_df)
            prediction_code = model.predict(processed_input)[0]
            suitability = 'Suitable' if prediction_code == 1 else 'Not Suitable'

            if suitability == 'Suitable':
                explanation_text = "*Could not generate rationale.*"
                if explainer and preprocessor_feature_names is not None:
                    try:
                        shap_values = explainer.shap_values(processed_input)
                        shap_values_instance = None
                        if isinstance(shap_values, np.ndarray) and shap_values.ndim >= 1:
                            shap_values_instance = shap_values[0]
                        if shap_values_instance is not None:
                            explanation_text = format_shap_explanation_user_focused(
                                shap_values_instance, preprocessor_feature_names, input_data, "Suitable", explanation_type='investment'
                            )
                        else: print(f"Warning: Unexpected SHAP format for {inv_type}.")
                    except Exception as shap_e: print(f"Inv Rec: SHAP failed for {inv_type}: {shap_e}")

                # --- *** ADD PROJECTION CALCULATION *** ---
                avg_annual_return = INVESTMENT_RETURN_MAPPING.get(details['Return']) # Get rate from mapping
                projected_value, total_growth = 0, 0 # Defaults
                if avg_annual_return is not None:
                    projected_value, total_growth = project_investment_growth(
                        projection_principal, avg_annual_return, projection_years
                    )
                # --- *** END PROJECTION CALCULATION *** ---

                recommendations.append({
                    "investment": inv_type,
                    "suitability": suitability,
                    "explanation": explanation_text,
                    # --- ADD PROJECTION RESULTS ---
                    "projected_value": projected_value,
                    "total_growth": total_growth,
                    "avg_annual_return_used": avg_annual_return * 100 if avg_annual_return is not None else "N/A" # For display
                })
        except Exception as e: print(f"Error processing investment {inv_type}: {e}"); traceback.print_exc()

    print(f"--- Finished Generating Investment Recs. Found {len(recommendations)} suitable. ---")
    if not recommendations: return [{"investment": "None Suitable", "explanation": "*Based on the analysis, no standard investments were deemed suitable.*"}]
    return recommendations

# ... (Keep get_risk_profile_and_explanation and get_planning_recommendation) ...