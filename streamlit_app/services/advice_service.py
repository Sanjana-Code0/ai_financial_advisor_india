# streamlit_app/services/advice_service.py
try:
    from . import db_service
    from ai_integration import prediction
except ImportError as e:
    print(f"CRITICAL ERROR importing modules within advice_service: {e}.")
    raise

STATIC_RISK_EXPLANATIONS = { # Keep your static explanations
    "Conservative": "This suggests you generally prefer safer options...",
    "Moderate": "This indicates you're comfortable taking some calculated risks...",
    "Aggressive": "This suggests you are comfortable with taking significant risks...",
    "Error": "Could not determine risk profile explanation.",
    "Default": "Your risk profile helps determine suitable investment strategies."
}

# *** MODIFIED function signature to accept projection parameters ***
def generate_advice(user_id: int, projection_principal_ui=100000, projection_years_ui=5):
    print(f"Generating advice for user_id: {user_id} with projection: P={projection_principal_ui}, Y={projection_years_ui}")
    profile_dict = db_service.get_profile(user_id)
    if not profile_dict: return {"error": "User profile not found."}

    profile_for_ai = profile_dict.copy()
    profile_for_ai.pop('id', None); profile_for_ai.pop('user_id', None)
    try: expected_risk_keys = prediction.RISK_FEATURE_ORDER
    except AttributeError: return {"error": "AI prediction component config error."}
    for key in expected_risk_keys: profile_for_ai.setdefault(key, None)

    risk_result_ai = prediction.get_risk_profile_and_explanation(profile_for_ai)
    if not risk_result_ai:
        ai_load_error = prediction.AI_COMPONENTS.get("load_error")
        error_msg = f"Could not generate risk assessment. {'AI components failed to load.' if ai_load_error else 'AI model error.'}"
        return {"error": error_msg}

    predicted_risk_profile = risk_result_ai.get('prediction', 'Error')
    risk_explanation_static = STATIC_RISK_EXPLANATIONS.get(predicted_risk_profile, STATIC_RISK_EXPLANATIONS["Default"])
    risk_explanation_detailed_shap = risk_result_ai.get('explanation', '*Detailed factor analysis unavailable.*')

    investment_recommendations = []
    planning_recommendation = {"actions": ["N/A"], "explanation": "Planning requires valid risk profile."}

    if predicted_risk_profile and predicted_risk_profile != 'Error':
        # *** PASS THE PROJECTION PARAMETERS FROM UI TO PREDICTION FUNCTION ***
        investment_recommendations = prediction.get_investment_recommendations_and_explanation(
            user_profile_dict_full=profile_for_ai,
            user_risk_profile=predicted_risk_profile,
            projection_principal=projection_principal_ui, # Pass UI value
            projection_years=projection_years_ui         # Pass UI value
        )
        suitable_investments_list = [rec for rec in investment_recommendations if rec.get('suitability') == 'Suitable']
        planning_recommendation = prediction.get_planning_recommendation(profile_for_ai, predicted_risk_profile, suitable_investments_list)
    else:
        investment_recommendations = [{"investment": "N/A", "explanation": "Cannot generate without valid risk profile."}]

    final_advice = {
        "risk_profile": predicted_risk_profile,
        "risk_explanation_simple": risk_explanation_static,
        "risk_explanation_detailed_shap": risk_explanation_detailed_shap,
        "investment_recommendations": investment_recommendations,
        "planning_recommendation": planning_recommendation
    }
    print(f"Advice generated successfully for user_id: {user_id}")
    return final_advice