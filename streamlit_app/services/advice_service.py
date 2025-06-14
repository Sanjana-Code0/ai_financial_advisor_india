# streamlit_app/services/advice_service.py
# (Ensure imports are correct: relative for db_service, standard for prediction)
try:
    from . import db_service
    from ai_integration import prediction # Relies on sys.path from calling page
except ImportError as e:
    print(f"CRITICAL ERROR importing modules within advice_service: {e}.")
    raise

# --- Static Risk Profile Explanations (Keep this dictionary) ---
STATIC_RISK_EXPLANATIONS = {
    "Conservative": "This suggests you generally prefer safer options for your money, even if it means slower growth. You likely prioritize protecting your initial investment over taking big chances for potentially high returns. This approach is often suitable for short-term goals or those nearing retirement.",
    "Moderate": "This indicates you're comfortable taking some calculated risks for potentially better returns than very safe options. You're looking for a balance – aiming for good growth over time without taking extreme chances. This approach often fits medium-to-long-term goals.",
    "Aggressive": "This suggests you are comfortable with taking significant risks for the chance of achieving higher long-term growth. You understand that investments might go up and down quite a bit (volatility), but you're focused on the potential for greater rewards over a longer period. This is often suitable for long-term goals where you have time to ride out market fluctuations.",
    "Error": "Could not determine risk profile explanation due to an error.",
    "Default": "Your risk profile helps determine suitable investment strategies."
}

def generate_advice(user_id: int):
    print(f"Generating advice for user_id: {user_id}")
    profile_dict = db_service.get_profile(user_id)
    if not profile_dict: return {"error": "User profile not found."}

    profile_for_ai = profile_dict.copy()
    profile_for_ai.pop('id', None); profile_for_ai.pop('user_id', None)
    try: expected_keys = prediction.RISK_FEATURE_ORDER
    except AttributeError: return {"error": "AI prediction component config error."}
    for key in expected_keys: profile_for_ai.setdefault(key, None)

    risk_result_ai = prediction.get_risk_profile_and_explanation(profile_for_ai)
    if not risk_result_ai:
        ai_load_error = prediction.AI_COMPONENTS.get("load_error")
        error_msg = f"Could not generate risk assessment. {'AI components failed to load.' if ai_load_error else 'AI model error.'}"
        return {"error": error_msg}

    predicted_risk_profile = risk_result_ai.get('prediction', 'Error')
    # This is the DETAILED SHAP explanation from prediction.py
    risk_explanation_detailed_shap = risk_result_ai.get('explanation', '*Detailed factor analysis unavailable.*')
    # This is the simple STATIC paragraph
    risk_explanation_static = STATIC_RISK_EXPLANATIONS.get(predicted_risk_profile, STATIC_RISK_EXPLANATIONS["Default"])

    investment_recommendations = []
    planning_recommendation = {"actions": ["N/A"], "explanation": "Planning requires valid risk profile."}

    if predicted_risk_profile and predicted_risk_profile != 'Error':
        investment_recommendations = prediction.get_investment_recommendations_and_explanation(predicted_risk_profile)
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