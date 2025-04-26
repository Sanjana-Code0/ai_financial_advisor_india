# streamlit_app/ai_integration/prediction.py
# (Keep imports and configuration at the top)
import joblib, pandas as pd, numpy as np, shap, os, streamlit as st, xgboost as xgb, traceback
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); MODEL_DIR = os.path.join(PROJECT_ROOT_DIR, 'models')
print(f"Attempting to load models from: {MODEL_DIR}")
RISK_PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'user_data_preprocessor.joblib'); RISK_MODEL_PATH = os.path.join(MODEL_DIR, 'risk_profile_rf_model.joblib')
INV_PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'investment_data_preprocessor.joblib'); INV_MODEL_PATH = os.path.join(MODEL_DIR, 'investment_suitability_xgb_model.joblib')
RISK_FEATURE_ORDER = ['AgeRange', 'IncomeRange', 'SavingsLevel', 'DebtLevel', 'HasDependents','PrimaryGoal', 'TimeHorizonYears', 'SelfReportedTolerance']
INV_FEATURE_ORDER = ['RiskProfile', 'InvestmentType', 'InvestmentVolRange', 'InvestmentRetRange']
AVAILABLE_INVESTMENTS = {'FD':{'Volatility':'Very Low','Return':'Very Low'},'PPF':{'Volatility':'Very Low','Return':'Low'},'DebtMF':{'Volatility':'Low','Return':'Low'},'IndexFund':{'Volatility':'Medium','Return':'Medium'},'BalancedMF':{'Volatility':'Medium','Return':'Medium'},'LargeCapMF':{'Volatility':'High','Return':'High'},'MidSmallCapMF':{'Volatility':'Very High','Return':'Very High'},'DirectEquity':{'Volatility':'Very High','Return':'Very High'}}

# --- *** MODIFIED: Load Models, Preprocessors, and SHAP Explainers (Corrected Syntax) *** ---
@st.cache_resource
def load_ai_components():
    """Loads all models, preprocessors, and initializes SHAP explainers."""
    print("Attempting to load ALL AI components via @st.cache_resource...")
    components = {
        "risk_preprocessor": None, "risk_model": None, "risk_explainer": None,
        "inv_preprocessor": None, "inv_model": None, "inv_explainer": None,
        "risk_feature_names": None, "inv_feature_names": None,
        "load_error": None
    }
    try:
        # --- Load Risk Components ---
        print(f"Checking for Risk Preprocessor at: {RISK_PREPROCESSOR_PATH}")
        if os.path.exists(RISK_PREPROCESSOR_PATH):
            components["risk_preprocessor"] = joblib.load(RISK_PREPROCESSOR_PATH)
            print("-> Risk preprocessor loaded.")
            # --- Get Risk Feature Names (in a separate try-except) ---
            try:
                components["risk_feature_names"] = components["risk_preprocessor"].get_feature_names_out()
                print(f"-> Risk feature names extracted ({len(components['risk_feature_names'])} features).")
            except Exception as e:
                 # Don't stop loading other things if this fails, just warn
                 print(f"Warning: Could not get feature names from risk preprocessor: {e}")
                 components["risk_feature_names"] = None
        else:
            # If a critical file is missing, record error and stop trying to load dependent parts
            raise FileNotFoundError(f"Risk preprocessor missing: {RISK_PREPROCESSOR_PATH}")

        print(f"Checking for Risk Model at: {RISK_MODEL_PATH}")
        if os.path.exists(RISK_MODEL_PATH):
            components["risk_model"] = joblib.load(RISK_MODEL_PATH)
            print("-> Risk model loaded.")
            # --- Initialize Risk Explainer ---
            if components["risk_model"]:
                try:
                    components["risk_explainer"] = shap.TreeExplainer(components["risk_model"])
                    print("-> SHAP explainer for risk model initialized.")
                except Exception as e:
                    print(f"Error initializing SHAP explainer for risk model: {e}")
                    components["risk_explainer"] = None # Set to None if init fails
        else:
             raise FileNotFoundError(f"Risk model missing: {RISK_MODEL_PATH}")

        # --- Load Investment Components ---
        print(f"Checking for Investment Preprocessor at: {INV_PREPROCESSOR_PATH}")
        if os.path.exists(INV_PREPROCESSOR_PATH):
            components["inv_preprocessor"] = joblib.load(INV_PREPROCESSOR_PATH)
            print("-> Investment preprocessor loaded.")
            # --- Get Investment Feature Names ---
            try:
                components["inv_feature_names"] = components["inv_preprocessor"].get_feature_names_out()
                print(f"-> Investment feature names extracted ({len(components['inv_feature_names'])} features).")
            except Exception as e:
                 print(f"Warning: Could not get feature names from investment preprocessor: {e}")
                 components["inv_feature_names"] = None
        else:
             raise FileNotFoundError(f"Investment preprocessor missing: {INV_PREPROCESSOR_PATH}")

        print(f"Checking for Investment Model at: {INV_MODEL_PATH}")
        if os.path.exists(INV_MODEL_PATH):
            components["inv_model"] = joblib.load(INV_MODEL_PATH)
            print("-> Investment model loaded.")
             # --- Initialize Investment Explainer ---
            if components["inv_model"]:
                try:
                    components["inv_explainer"] = shap.TreeExplainer(components["inv_model"])
                    print("-> SHAP explainer for investment model initialized.")
                except Exception as e:
                    print(f"Error initializing SHAP explainer for investment model: {e}")
                    components["inv_explainer"] = None
        else:
             raise FileNotFoundError(f"Investment model missing: {INV_MODEL_PATH}")

        print("--- AI component loading process finished successfully. ---")

    # Catch specific critical errors like missing files
    except FileNotFoundError as e:
        critical_error = f"AI Component Loading Error: Required file not found - {e}"
        print(f"!!! {critical_error} !!!")
        components["load_error"] = str(e) # Store just the error message
        st.error(critical_error) # Show error in UI too
    # Catch any other unexpected errors during loading
    except Exception as e:
        critical_error = f"A critical error occurred during AI component loading: {e}"
        print(critical_error)
        traceback.print_exc()
        components["load_error"] = str(e)
        st.error(critical_error)

    return components

# Load components globally
AI_COMPONENTS = load_ai_components()

# --- Helper Function to Format SHAP Explanation (Keep improved version) ---
def format_shap_explanation_detailed(shap_values_instance, feature_names, max_features=5):
    # ...(Keep the detailed formatter function as provided in the previous step)...
    intro = "**Key factors influencing this assessment:**\n\n"
    no_detail_msg = "*Detailed factor analysis unavailable.*"
    print(f"--- format_shap_explanation_detailed called ---")
    try:
        if feature_names is None: print("DEBUG: format_shap - Feature names None."); return no_detail_msg
        if shap_values_instance is None: print("DEBUG: format_shap - SHAP values None."); return no_detail_msg
        if len(feature_names)==0 or len(shap_values_instance)==0: print("DEBUG: format_shap - Features/SHAP empty."); return no_detail_msg
        print(f"DEBUG: format_shap - Received {len(feature_names)} names, {len(shap_values_instance)} values.")
        if len(feature_names) != len(shap_values_instance):
             print(f"ERROR: format_shap - Mismatch length ({len(feature_names)} vs {len(shap_values_instance)}).")
             min_len = min(len(feature_names), len(shap_values_instance));
             if min_len==0: return "Explanation error (zero length)."
             print(f"Warning: format_shap - Truncating to {min_len}."); feature_names=feature_names[:min_len]; shap_values_instance=shap_values_instance[:min_len]
        print("DEBUG: format_shap - Creating DataFrame...")
        feature_shap = pd.DataFrame({'feature': feature_names, 'shap_value': shap_values_instance})
        feature_shap['abs_shap'] = np.abs(feature_shap['shap_value'])
        feature_shap = feature_shap.sort_values(by='abs_shap', ascending=False).reset_index(drop=True)
        print(f"DEBUG: format_shap - Top 5 features before filtering:\n{feature_shap.head()}")
        feature_shap = feature_shap[feature_shap['abs_shap'] > 0.01]
        print(f"DEBUG: format_shap - Top {max_features} features after filtering:\n{feature_shap.head(max_features)}")
        if feature_shap.empty: print("DEBUG: format_shap - No significant factors."); return "*No factors significantly influenced prediction.*"
        explanation_points = []
        for i, row in enumerate(feature_shap.head(max_features).itertuples()):
            feature_name_raw = row.feature; shap_value = row.shap_value
            print(f"DEBUG: format_shap - Processing feature: {feature_name_raw}, SHAP: {shap_value:.4f}")
            feature_parts = feature_name_raw.split('__'); feature_desc = feature_parts[-1].replace('_', ' ')
            if len(feature_parts) > 1 and feature_parts[0] == 'cat':
                 category_name = feature_parts[1].split('_')[0]; value = feature_desc.replace(category_name, '', 1).strip()
                 feature_display = f"{category_name} = '{value}'"
            elif len(feature_parts) > 1 and feature_parts[0] == 'num': feature_display = feature_desc.capitalize()
            else: feature_display = feature_desc.capitalize()
            contribution = "increased the likelihood" if shap_value > 0 else "decreased the likelihood"
            strength = "significantly" if abs(shap_value) > 0.2 else "moderately" if abs(shap_value) > 0.05 else "slightly"
            point = f"*   **{feature_display}:** {strength} {contribution} of this outcome (Impact Score: {shap_value:+.3f})."
            explanation_points.append(point)
        print("DEBUG: format_shap - Formatting successful.")
        return intro + "\n".join(explanation_points)
    except Exception as e: print(f"ERROR in format_shap_explanation_detailed: {e}"); traceback.print_exc(); return no_detail_msg

# --- Risk Profile Prediction Function (Keep restored SHAP) ---
def get_risk_profile_and_explanation(user_profile_dict):
    # ...(Keep the function logic exactly as in the previous step)...
    print("--- Running Risk Prediction (with SHAP) ---")
    preprocessor = AI_COMPONENTS.get("risk_preprocessor"); model = AI_COMPONENTS.get("risk_model"); explainer = AI_COMPONENTS.get("risk_explainer"); feature_names = AI_COMPONENTS.get("risk_feature_names"); load_error = AI_COMPONENTS.get("load_error")
    if load_error or not all([preprocessor, model]): error_msg = f"Risk components not loaded. Error: {load_error or 'Component missing'}"; print(error_msg); return None
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
        explanation_text = "*Detailed explanation requires SHAP.*"
        if explainer and feature_names is not None:
            try:
                print(f"Risk Pred: Calculating SHAP values...")
                shap_values = explainer.shap_values(processed_input)
                shap_values_instance = None
                if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                     if 0 <= predicted_class_index < shap_values.shape[2]: shap_values_instance = shap_values[0, :, predicted_class_index]
                     else: print(f"Error: Invalid index {predicted_class_index} for SHAP array.")
                elif isinstance(shap_values, list) and len(shap_values) == len(classes):
                     if 0 <= predicted_class_index < len(shap_values): shap_values_instance = shap_values[predicted_class_index][0]
                     else: print(f"Error: Invalid index {predicted_class_index} for SHAP list.")
                else: print(f"Warning: Unexpected SHAP format.")
                if shap_values_instance is not None:
                     print(f"Risk Pred: Extracted SHAP values. Shape: {shap_values_instance.shape}. Feature names count: {len(feature_names)}")
                     explanation_text = format_shap_explanation_detailed(shap_values_instance, feature_names)
                else: explanation_text = "*Could not process explanation format.*"
            except Exception as shap_e: print(f"Risk Pred: SHAP calculation failed: {shap_e}"); traceback.print_exc(); explanation_text = "*Error generating SHAP explanation.*"
        elif not explainer: explanation_text = "*Explanation unavailable (explainer).*"
        else: explanation_text = "*Explanation unavailable (feature names).*"
        print(f"--- Risk Prediction Finished: {prediction_label} ---")
        return {'prediction': str(prediction_label), 'explanation': explanation_text}
    except Exception as e: error_msg = f"Error during risk prediction: {e}"; print(error_msg); traceback.print_exc(); st.error(error_msg); return None

# --- Investment Recommendation Function (Keep using detailed formatter) ---
def get_investment_recommendations_and_explanation(user_risk_profile: str):
     # ...(Keep the function logic exactly as in the previous step)...
    print(f"\n--- Running Investment Recommendations for Profile: {user_risk_profile} ---")
    preprocessor=AI_COMPONENTS.get("inv_preprocessor"); model=AI_COMPONENTS.get("inv_model"); explainer=AI_COMPONENTS.get("inv_explainer"); feature_names=AI_COMPONENTS.get("inv_feature_names"); load_error=AI_COMPONENTS.get("load_error")
    if load_error or not all([preprocessor,model]): return [{"investment": "Error", "explanation": "AI components missing."}]
    recommendations = []
    for inv_type, details in AVAILABLE_INVESTMENTS.items():
        try:
            input_data={'RiskProfile':user_risk_profile,'InvestmentType':inv_type,'InvestmentVolRange':details['Volatility'],'InvestmentRetRange':details['Return']}
            input_df=pd.DataFrame([input_data],columns=INV_FEATURE_ORDER); processed_input=preprocessor.transform(input_df)
            prediction_code=model.predict(processed_input)[0]; suitability='Suitable' if prediction_code==1 else 'Not Suitable'
            if suitability == 'Suitable':
                explanation_text="*Detailed rationale unavailable.*"
                if explainer and feature_names is not None:
                    try:
                        print(f"Inv Rec: Calculating SHAP for {inv_type}..."); shap_values=explainer.shap_values(processed_input)
                        shap_values_instance = None
                        if isinstance(shap_values, np.ndarray):
                            if shap_values.ndim >= 1 : shap_values_instance = shap_values[0]
                        else: print(f"Warning: Unexpected SHAP format for {inv_type}.")
                        if shap_values_instance is not None:
                             print(f"Inv Rec: Extracted SHAP values for {inv_type}. Shape: {shap_values_instance.shape}. Feature names count: {len(feature_names)}")
                             explanation_text = format_shap_explanation_detailed(shap_values_instance, feature_names) # Call detailed formatter
                        else: explanation_text = "*Could not process explanation format.*"
                    except Exception as shap_e: print(f"Inv Rec: SHAP failed for {inv_type}: {shap_e}"); traceback.print_exc(); explanation_text = "*Error generating rationale.*"
                elif not explainer: explanation_text = "*Rationale unavailable (explainer).*"
                else: explanation_text = "*Rationale unavailable (feature names).*"
                recommendations.append({"investment":inv_type,"suitability":suitability,"explanation":explanation_text})
        except Exception as e: error_msg=f"Error predicting suitability for {inv_type}: {e}"; print(error_msg); traceback.print_exc()
    print(f"--- Finished Generating Investment Recs. Found {len(recommendations)} suitable. ---")
    if not recommendations: return [{"investment":"None Suitable","explanation":"*No standard investments deemed suitable.*"}]
    return recommendations

# --- Placeholder Function for RL Planning (Keep as is) ---
def get_planning_recommendation(user_profile_dict, risk_profile, suitable_investments):
    # ...(placeholder logic remains the same)...
    print(f"--- Generating Planning Recommendation (Placeholder) ---")
    plan_actions = [f"Suggested Monthly Savings: 15% of income (Placeholder)", f"Investment Allocation: 60% IndexFund, 40% LargeCapMF (Placeholder)"]
    plan_explanation = f"This placeholder plan balances growth for your '{user_profile_dict.get('PrimaryGoal')}' goal with your '{risk_profile}' profile. (Reinforcement Learning agent integration pending)."
    print("--- Planning Recommendation Placeholder Generated ---")
    return {"actions": plan_actions, "explanation": plan_explanation}