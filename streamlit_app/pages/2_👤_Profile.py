# streamlit_app/pages/2_👤_Profile.py
# --- (Keep sys.path block and initial imports) ---
import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'));
if project_root not in sys.path: sys.path.insert(0, project_root)
import streamlit as st, time, traceback
try: from services import db_service; from utils import load_css
except ImportError as e: st.error(f"Failed to import modules: {e}."); st.stop()

load_css("style.css")
st.header("👤 Your Financial Profile")
st.write("Keep this updated for the most relevant financial insights.")
st.markdown("---")
if not st.session_state.get('logged_in'): st.warning("Please login first..."); st.stop()
user_id = st.session_state.get('user_id')
if not user_id: st.error("Error: User ID not found."); st.stop()
st.info("ℹ️ Fill out this profile accurately. Your answers help tailor the advice.")

existing_profile_dict = db_service.get_profile(user_id)
if existing_profile_dict is None: st.error("Could not load profile data."); existing_profile_dict = {}
elif not existing_profile_dict: st.caption("No profile found. Fill out the form.")
else: st.caption("Existing profile data loaded.")

# Define Profile Questions Options
age_ranges=['18-24','25-34','35-44','45-54','55+']; income_ranges=['< ₹5 LPA','₹5-12 LPA','₹12-25 LPA','₹25+ LPA']
savings_levels=['Low','Medium','High']; debt_levels=['Low','Medium','High']; dependents_options=['Yes','No']
goal_options=['Retirement','ChildEdu','Property','Marriage','Business','Wealth','Other']
time_horizon_ranges=['< 5 years','5 - 10 years','11 - 15 years','16 - 20 years','More than 20 years']
time_horizon_map={'< 5 years':3,'5 - 10 years':7,'11 - 15 years':13,'16 - 20 years':18,'More than 20 years':25}
tolerance_levels=['Low','Medium','High']
# --- NEW OPTIONS ---
investment_knowledge_levels = ['Beginner', 'Intermediate', 'Advanced']
liquidity_needs_levels = ['Low', 'Medium', 'High']

# Helper functions (keep as is)
def get_index(options_list, value): # ...
    if value is None: return 0
    try: return options_list.index(value)
    except ValueError: print(f"Warning: Value '{value}' not in {options_list}."); return 0
def get_range_from_years(year_map, years_value): # ...
    default_range = list(year_map.keys())[0]
    if years_value is None: return default_range
    for key, val in year_map.items():
        if val == years_value: return key
    print(f"Warning: Years value '{years_value}' not found."); return default_range


with st.container():
    with st.form("profile_form"):
        st.subheader("🧑 About You"); col1, col2 = st.columns(2)
        with col1: age_range = st.selectbox("Age Range", age_ranges, index=get_index(age_ranges, existing_profile_dict.get("AgeRange")))
        with col2: has_dependents = st.radio("Financially Support Dependents?", dependents_options, index=get_index(dependents_options, existing_profile_dict.get("HasDependents")), horizontal=True)
        st.markdown("---")

        st.subheader("💰 Financial Situation"); col1, col2 = st.columns(2)
        with col1:
            income_range = st.selectbox("Annual Household Income (INR)", income_ranges, index=get_index(income_ranges, existing_profile_dict.get("IncomeRange")))
            savings_level = st.selectbox("Savings Level", savings_levels, index=get_index(savings_levels, existing_profile_dict.get("SavingsLevel")), help="Vs. monthly expenses (Low: <3mo, Med: 3-6mo, High: >6mo)")
        with col2:
            debt_level = st.selectbox("Debt Level (Non-Mortgage)", debt_levels, index=get_index(debt_levels, existing_profile_dict.get("DebtLevel")), help="Relative to income (Low: Minimal, Med: Budgeted, High: Significant)")
        st.markdown("---")

        st.subheader("🎯 Goals & Investment Style") # Combined subheader
        col1, col2, col3 = st.columns(3) # Three columns
        with col1: primary_goal = st.selectbox("Primary Financial Goal", goal_options, index=get_index(goal_options, existing_profile_dict.get("PrimaryGoal")))
        with col2: time_horizon_range = st.select_slider("Time Horizon for Goal", options=time_horizon_ranges, value=get_range_from_years(time_horizon_map, existing_profile_dict.get("TimeHorizonYears")))
        with col3: # New field
            investment_knowledge = st.selectbox("Investment Knowledge", investment_knowledge_levels, index=get_index(investment_knowledge_levels, existing_profile_dict.get("InvestmentKnowledge")), help="Your understanding of investments.")
        st.markdown("---")

        st.subheader("🎢 Investment Comfort & Needs")
        col1, col2 = st.columns(2)
        with col1:
            self_reported_tolerance = st.radio("Investment Risk Tolerance", options=tolerance_levels, index=get_index(tolerance_levels, existing_profile_dict.get("SelfReportedTolerance")), horizontal=True, help="Low: Safety first, Med: Balanced, High: Higher growth/risk")
        with col2: # New field
            liquidity_needs = st.selectbox("Liquidity Needs", liquidity_needs_levels, index=get_index(liquidity_needs_levels, existing_profile_dict.get("LiquidityNeeds")), help="How soon might you need to access these funds? (Low: Long-term lock-in okay, High: May need access soon)")
        st.markdown("---")

        submitted = st.form_submit_button("💾 Save / Update Profile")

if submitted:
    with st.spinner("⏳ Saving profile..."):
        time_horizon_years = time_horizon_map[time_horizon_range]
        profile_data = {
            "AgeRange": age_range, "IncomeRange": income_range, "SavingsLevel": savings_level,
            "DebtLevel": debt_level, "HasDependents": has_dependents, "PrimaryGoal": primary_goal,
            "TimeHorizonYears": time_horizon_years, "SelfReportedTolerance": self_reported_tolerance,
            # --- ADD NEW FIELDS TO SAVE ---
            "InvestmentKnowledge": investment_knowledge,
            "LiquidityNeeds": liquidity_needs
        }
        try:
            saved_profile = db_service.save_or_update_profile(user_id, profile_data)
            if saved_profile: st.success("✅ Profile saved successfully!"); time.sleep(1.5); st.rerun()
            else: st.error("❌ Failed to save profile. Please check logs.")
        except Exception as e: st.error(f"❌ Failed to save profile: Error occurred."); print(f"Error saving profile: {e}"); traceback.print_exc()