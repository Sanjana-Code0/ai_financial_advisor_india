# streamlit_app/ai_integration/prediction.py
# (Keep imports and all other code at the top: PROJECT_ROOT_DIR, MODEL_DIR, paths, feature orders, AVAILABLE_INVESTMENTS, load_ai_components)
# The functions get_risk_profile_and_explanation and get_investment_recommendations_and_explanation
# will now call this new formatter.

import joblib, pandas as pd, numpy as np, shap, os, streamlit as st, xgboost as xgb, traceback
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); MODEL_DIR = os.path.join(PROJECT_ROOT_DIR, 'models')
RISK_PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'user_data_preprocessor.joblib'); RISK_MODEL_PATH = os.path.join(MODEL_DIR, 'risk_profile_rf_model.joblib')
INV_PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'investment_data_preprocessor.joblib'); INV_MODEL_PATH = os.path.join(MODEL_DIR, 'investment_suitability_xgb_model.joblib')
RISK_FEATURE_ORDER = ['AgeRange', 'IncomeRange', 'SavingsLevel', 'DebtLevel', 'HasDependents','PrimaryGoal', 'TimeHorizonYears', 'SelfReportedTolerance']
INV_FEATURE_ORDER = ['RiskProfile', 'InvestmentType', 'InvestmentVolRange', 'InvestmentRetRange']
AVAILABLE_INVESTMENTS = {'FD':{'Volatility':'Very Low','Return':'Very Low'},'PPF':{'Volatility':'Very Low','Return':'Low'},'DebtMF':{'Volatility':'Low','Return':'Low'},'IndexFund':{'Volatility':'Medium','Return':'Medium'},'BalancedMF':{'Volatility':'Medium','Return':'Medium'},'LargeCapMF':{'Volatility':'High','Return':'High'},'MidSmallCapMF':{'Volatility':'Very High','Return':'Very High'},'DirectEquity':{'Volatility':'Very High','Return':'Very High'}}
@st.cache_resource
def load_ai_components():
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


# --- *** NEW USER-FOCUSED SHAP EXPLANATION FORMATTER *** ---
def format_shap_explanation_user_focused(shap_values_instance, preprocessor_feature_names, original_input_dict, predicted_outcome_label, explanation_type='risk', max_features=3):
    """
    Generates simpler, user-friendly explanation text, focusing on how the user's
    specific inputs influenced the decision, rather than all possible one-hot encoded features.
    """
    if explanation_type == 'risk':
        intro = f"Here's what primarily led to the **'{predicted_outcome_label}'** risk profile assessment:\n\n"
    elif explanation_type == 'investment':
        intro = f"Here's why this investment is considered **'{predicted_outcome_label}'** for you:\n\n"
    else:
        intro = "**Key factors in this decision:**\n\n"

    no_detail_msg = "*We considered your overall profile, but a detailed breakdown of specific factors is currently unavailable.*"
    print(f"--- format_shap_explanation_user_focused for {explanation_type} ---")

    try:
        # Initial checks
        if preprocessor_feature_names is None or shap_values_instance is None or \
           len(preprocessor_feature_names) == 0 or len(shap_values_instance) == 0 or original_input_dict is None:
            print("DEBUG: format_user_focused - Missing preprocessor_feature_names, SHAP values, or original_input_dict.")
            return no_detail_msg
        if len(preprocessor_feature_names) != len(shap_values_instance):
            print(f"ERROR: format_user_focused - Mismatch length ({len(preprocessor_feature_names)} vs {len(shap_values_instance)}).")
            return "Explanation error (internal mismatch)."

        # Map SHAP values to original features by identifying which one-hot encoded
        # feature corresponds to the user's actual input.
        contributions = {} # Store as {original_feature_name: shap_value}

        for i, full_feature_name in enumerate(preprocessor_feature_names):
            parts = full_feature_name.split('__') # e.g., "cat__AgeRange_18-24" or "num__TimeHorizonYears"
            if len(parts) < 2: # Not a ColumnTransformer output or unexpected format
                # This might be a feature that was passed through or not transformed as expected
                # For now, we'll consider it if it's in original_input_dict
                if full_feature_name in original_input_dict:
                    contributions[full_feature_name] = contributions.get(full_feature_name, 0) + shap_values_instance[i]
                continue

            transformer_type = parts[0] # 'cat' or 'num'
            original_feature_name_with_suffix = parts[1] # e.g., 'AgeRange_18-24' or 'TimeHorizonYears'

            # For categorical features, check if this encoded feature matches user's input
            if transformer_type == 'cat':
                # Extract base name, e.g., 'AgeRange' from 'AgeRange_18-24'
                # This assumes the suffix is the value after the last underscore
                # or that the part before the first underscore is the base if no other underscores
                split_suffix = original_feature_name_with_suffix.rsplit('_', 1)
                base_feature_name = split_suffix[0]
                encoded_value_part = split_suffix[1] if len(split_suffix) > 1 else original_feature_name_with_suffix

                if base_feature_name in original_input_dict:
                    user_actual_value = str(original_input_dict[base_feature_name])
                    if user_actual_value == encoded_value_part:
                        # This specific one-hot column corresponds to the user's actual input
                        contributions[base_feature_name] = contributions.get(base_feature_name, 0) + shap_values_instance[i]
            elif transformer_type == 'num':
                # For numerical, the name is usually direct (e.g., 'TimeHorizonYears')
                base_feature_name = original_feature_name_with_suffix
                if base_feature_name in original_input_dict:
                     contributions[base_feature_name] = contributions.get(base_feature_name, 0) + shap_values_instance[i]


        if not contributions:
            print("DEBUG: format_user_focused - No original features could be mapped from SHAP contributions.")
            return no_detail_msg

        # Create DataFrame from aggregated contributions to original features
        df_contributions = pd.DataFrame(list(contributions.items()), columns=['original_feature', 'total_shap_value'])
        df_contributions['abs_shap'] = np.abs(df_contributions['total_shap_value'])
        df_contributions = df_contributions.sort_values(by='abs_shap', ascending=False).reset_index(drop=True)
        # Optional: Filter very small contributions
        df_contributions = df_contributions[df_contributions['abs_shap'] > 0.01]


        if df_contributions.empty:
            print("DEBUG: format_user_focused - No significant user input factors after filtering.")
            return "*Your overall profile was assessed, but no single input stood out significantly.*"

        explanation_points = []
        for i, row in enumerate(df_contributions.head(max_features).itertuples()):
            original_feature = row.original_feature
            user_value = original_input_dict.get(original_feature) # Get the user's actual input
            shap_value = row.total_shap_value

            friendly_name_map = {
                'AgeRange': 'Your age group', 'IncomeRange': 'Your income level',
                'SavingsLevel': 'Your savings level', 'DebtLevel': 'Your debt level',
                'HasDependents': 'Whether you have dependents', 'PrimaryGoal': 'Your primary goal',
                'TimeHorizonYears': 'Your investment time horizon', 'SelfReportedTolerance': 'Your stated risk comfort',
                'RiskProfile': "Your overall risk profile", # For investment context
                'InvestmentType': "The type of investment", # For investment context
                'InvestmentVolRange': "The investment's typical volatility level", # For investment context
                'InvestmentRetRange': "The investment's potential return category" # For investment context
            }
            friendly_name = friendly_name_map.get(original_feature, original_feature.replace('_', ' ').capitalize())
            value_display = f" (which is '{user_value}')" if user_value is not None else ""
            if original_feature == 'HasDependents': # Special handling for Yes/No
                value_display = f" (which is '{'Yes' if user_value == 'Yes' else 'No'}')"


            # Simplified interpretation based on SHAP value sign
            # Positive SHAP value means this feature (with its user-given value) pushed towards the predicted outcome
            # Negative SHAP value means it pushed away from the predicted outcome
            point = ""
            if explanation_type == 'risk':
                if shap_value > 0.02: # Threshold for "strong" influence
                    point = f"*   **{friendly_name}{value_display}** was a key factor supporting this '{predicted_outcome_label}' assessment."
                elif shap_value > 0.01: # Threshold for "moderate" influence
                    point = f"*   **{friendly_name}{value_display}** contributed to this '{predicted_outcome_label}' assessment."
                # Optionally, could add a line for negative impacts if desired for full transparency,
                # but for simplicity, focusing on positive contributions to the predicted class.
            elif explanation_type == 'investment':
                if shap_value > 0.02:
                     point = f"*   **{friendly_name}{value_display}** is a strong reason this investment is considered suitable."
                elif shap_value > 0.01:
                     point = f"*   **{friendly_name}{value_display}** supports the suitability of this investment."

            if point: # Only add if a meaningful point was constructed
                explanation_points.append(point)

        if not explanation_points:
             return "*While your overall profile was assessed, specific factors had a mixed or minor influence on this particular outcome.*"

        return intro + "\n".join(explanation_points)

    except Exception as e:
        print(f"ERROR in format_shap_explanation_user_focused: {e}"); traceback.print_exc()
        return no_detail_msg
# --- *** End of New User-Focused Formatter *** ---


# --- Risk Profile Prediction Function ---
def get_risk_profile_and_explanation(user_profile_dict):
    """Predicts risk profile label AND generates user-focused SHAP explanation."""
    # ... (Loading components and initial checks as before) ...
    print("--- Running Risk Prediction (with user-focused SHAP) ---")
    preprocessor = AI_COMPONENTS.get("risk_preprocessor"); model = AI_COMPONENTS.get("risk_model"); explainer = AI_COMPONENTS.get("risk_explainer"); preprocessor_feature_names = AI_COMPONENTS.get("risk_feature_names"); load_error = AI_COMPONENTS.get("load_error")
    if load_error or not all([preprocessor, model]): return None

    print(f"Risk Pred: Received profile data keys: {list(user_profile_dict.keys())}")
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
                     # *** CALL NEW USER-FOCUSED FORMATTER ***
                     explanation_text = format_shap_explanation_user_focused(
                         shap_values_instance, preprocessor_feature_names, user_profile_dict, prediction_label, explanation_type='risk'
                     )
                else: explanation_text = "*Could not process explanation format.*"
            except Exception as shap_e: print(f"Risk Pred: SHAP calculation failed: {shap_e}"); traceback.print_exc(); explanation_text = "*Error generating risk factors.*"
        elif not explainer: explanation_text = "*Explanation unavailable (explainer).*"
        else: explanation_text = "*Explanation unavailable (feature names).*"

        print(f"--- Risk Prediction Finished: {prediction_label} ---")
        return {'prediction': str(prediction_label), 'explanation': explanation_text}
    except Exception as e: error_msg = f"Error during risk prediction: {e}"; print(error_msg); traceback.print_exc(); st.error(error_msg); return None


# --- Investment Recommendation Function ---
def get_investment_recommendations_and_explanation(user_risk_profile: str):
    """Predicts suitability and generates USER-FOCUSED explanations for investments."""
    print(f"\n--- Running Investment Recommendations for Profile: {user_risk_profile} ---")
    preprocessor=AI_COMPONENTS.get("inv_preprocessor"); model=AI_COMPONENTS.get("inv_model")
    explainer=AI_COMPONENTS.get("inv_explainer"); preprocessor_feature_names=AI_COMPONENTS.get("inv_feature_names") # Use preprocessor's feature names
    load_error=AI_COMPONENTS.get("load_error")
    if load_error or not all([preprocessor,model]): return [{"investment": "Error", "explanation": "AI components missing."}]

    recommendations = []
    for inv_type, details in AVAILABLE_INVESTMENTS.items():
        try:
            # This input_data is the 'original_input_dict' for the investment model
            input_data={'RiskProfile':user_risk_profile,'InvestmentType':inv_type,'InvestmentVolRange':details['Volatility'],'InvestmentRetRange':details['Return']}
            input_df=pd.DataFrame([input_data],columns=INV_FEATURE_ORDER); processed_input=preprocessor.transform(input_df)
            prediction_code=model.predict(processed_input)[0]; suitability='Suitable' if prediction_code==1 else 'Not Suitable'

            if suitability == 'Suitable':
                explanation_text="*Detailed rationale unavailable.*"
                if explainer and preprocessor_feature_names is not None:
                    try:
                        print(f"Inv Rec: Calculating SHAP for {inv_type}..."); shap_values=explainer.shap_values(processed_input)
                        shap_values_instance = None
                        if isinstance(shap_values, np.ndarray):
                            if shap_values.ndim >= 1 : shap_values_instance = shap_values[0]
                        else: print(f"Warning: Unexpected SHAP format for {inv_type}.")

                        if shap_values_instance is not None:
                             print(f"Inv Rec: Extracted SHAP values for {inv_type}. Shape: {shap_values_instance.shape}.")
                             # *** CALL NEW USER-FOCUSED FORMATTER ***
                             explanation_text = format_shap_explanation_user_focused(
                                 shap_values_instance, preprocessor_feature_names, input_data, "Suitable", explanation_type='investment'
                             )
                        else: explanation_text = "*Could not process explanation format.*"
                    except Exception as shap_e: print(f"Inv Rec: SHAP failed for {inv_type}: {shap_e}"); traceback.print_exc(); explanation_text = "*Error generating rationale.*"
                elif not explainer: explanation_text = "*Rationale unavailable (explainer).*"
                else: explanation_text = "*Rationale unavailable (feature names).*"
                recommendations.append({"investment":inv_type,"suitability":suitability,"explanation":explanation_text})
        except Exception as e: error_msg=f"Error predicting suitability for {inv_type}: {e}"; print(error_msg); traceback.print_exc()

    print(f"--- Finished Generating Investment Recs. Found {len(recommendations)} suitable. ---")
    if not recommendations: return [{"investment":"None Suitable","explanation":"*Based on the analysis, no standard investments were deemed suitable.*"}]
    return recommendations


# --- Placeholder Function for RL Planning (Keep as is) ---
def get_planning_recommendation(user_profile_dict, risk_profile, suitable_investments):
    # ...(placeholder logic remains the same)...
    print(f"--- Generating Planning Recommendation (Placeholder) ---")
    plan_actions = [f"Suggested Monthly Savings: 15% of income (Placeholder)", f"Investment Allocation: 60% IndexFund, 40% LargeCapMF (Placeholder)"]
    plan_explanation = f"This placeholder plan balances growth for your '{user_profile_dict.get('PrimaryGoal')}' goal with your '{risk_profile}' profile. (Reinforcement Learning agent integration pending)."
    print("--- Planning Recommendation Placeholder Generated ---")
    return {"actions": plan_actions, "explanation": plan_explanation}