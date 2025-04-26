# streamlit_app/pages/3_📊_Dashboard_Advice.py
# --- Add sys.path block at the VERY TOP ---
import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)
# --- End of block ---

import streamlit as st
try:
    from services import advice_service, db_service
    from utils import load_css # Import CSS loader
except ImportError as e: st.error(f"Failed to import modules: {e}."); st.stop()

# --- Load CSS ---
load_css("style.css")

# NO st.set_page_config() here
st.header("📊 Dashboard & Financial Advice")
st.write("Your personalized financial snapshot and guidance.")
st.markdown("---")

# Check Login Status/Profile Complete (Keep as before)
if not st.session_state.get('logged_in'): st.warning("Please login first..."); st.stop()
user_id = st.session_state.get('user_id');
if not user_id: st.error("Error: User ID not found."); st.stop()
profile_complete = db_service.is_profile_complete(user_id)
if not profile_complete: st.warning("Please complete your profile first..."); st.stop()

st.info(f"👋 Welcome, **{st.session_state.get('username', 'User')}**!")

# Button to Get Advice
if st.button("🚀 Get My Financial Advice"):
    st.markdown("---")
    with st.spinner("⏳ Analyzing profile & generating advice..."):
        advice_result = advice_service.generate_advice(user_id)

        if advice_result and "error" not in advice_result:
            # --- Display Financial Assessment ---
            with st.container(): # Use container for styling
                st.subheader("📈 Your Financial Assessment")
                col1, col2 = st.columns(2)
                risk_profile = advice_result.get("risk_profile", "N/A")
                with col1: st.metric(label="Predicted Risk Profile", value=risk_profile)
                with col2:
                     profile_info = db_service.get_profile(user_id)
                     st.metric(label="Primary Goal", value=profile_info.get("PrimaryGoal", "N/A") if profile_info else "N/A")

                # --- *** MODIFIED: Display BOTH Risk Explanations *** ---
                st.markdown("**Understanding Your Risk Profile:**")
                # Display Static Explanation First
                st.write(advice_result.get("risk_explanation_simple", "*Explanation not available.*"))
                st.markdown("---") # Separator
                # Display Detailed SHAP Explanation in an Expander
                with st.expander("View Detailed Factors (AI Analysis)"):
                    st.markdown(advice_result.get("risk_explanation_detailed", "*Detailed factor analysis unavailable.*"))
                # --- *** End of Modification *** ---

            st.markdown("---") # Separator between sections

            # --- Display Investment Recommendations (Keep as before) ---
            with st.container():
                st.subheader("💡 Investment Recommendations")
                investment_recs = advice_result.get("investment_recommendations", [])
                if not investment_recs: st.info("*No specific investment recommendations generated.*")
                elif investment_recs[0].get("investment") == "Error": st.warning(f"*Could not generate recommendations: {investment_recs[0].get('explanation')}*")
                elif investment_recs[0].get("investment") == "None Suitable": st.info(investment_recs[0].get("explanation"))
                else:
                    st.write("**Based on your risk profile, these approaches might be suitable:**")
                    num_cols = 2; cols = st.columns(num_cols); col_idx = 0
                    for rec in investment_recs:
                        with cols[col_idx % num_cols]:
                            st.markdown(f"##### **{rec.get('investment', 'Unknown')}**")
                            with st.expander("View Rationale", expanded=False): # Keep rationale collapsed by default
                                st.markdown(rec.get('explanation', '*Detailed explanation not available.*'))
                        col_idx += 1

            st.markdown("---") # Separator between sections

            # --- Display Planning Recommendation Section (Keep as before) ---
            with st.container():
                st.subheader("🧭 Personalized Planning Actions (Placeholder)")
                planning_rec = advice_result.get("planning_recommendation")
                if planning_rec:
                     # Keep expander collapsed by default maybe
                     with st.expander("View Suggested Actions & Rationale", expanded=False):
                         st.markdown("**Suggested Actions:**")
                         action_list = planning_rec.get("actions", ["*Not available*"])
                         if action_list:
                             for action_item in action_list: st.markdown(f"- {action_item}")
                         else: st.markdown("*No specific actions suggested.*")
                         st.markdown("\n**Rationale:**")
                         st.markdown(planning_rec.get("explanation", "*Explanation not available.*"))
                else: st.info("*Planning recommendations are currently unavailable.*")

            st.markdown("---")
            st.success("✅ Advice generated successfully!")

        elif advice_result and "error" in advice_result: st.error(f"❌ Could not generate advice: {advice_result['error']}")
        else: st.error("❌ An unexpected error occurred.")

else:
    st.markdown("Click the button above to get your latest financial assessment.")

st.markdown("---")
st.caption("Disclaimer: Suggestions only. Consult a professional financial advisor.")