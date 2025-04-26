# --- Add sys.path block at the VERY TOP ---
import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)
# --- End of block ---

import streamlit as st
import time # <<<<<<<<<<< ADD THIS IMPORT
try: from services import db_service; from utils import load_css
except ImportError as e: st.error(f"Failed to import modules: {e}."); st.stop()
import pandas as pd
import traceback

# --- Load CSS ---
load_css("style.css")

# NO st.set_page_config() here
st.header("👤 Your Financial Profile")
st.write("Keep this updated for the most relevant financial insights.")
st.markdown("---")

# Check Login Status
if not st.session_state.get('logged_in'): st.warning("Please login first..."); st.stop()
user_id = st.session_state.get('user_id')
if not user_id: st.error("Error: User ID not found."); st.stop()

st.info("ℹ️ Fill out this profile accurately. Your answers help tailor the advice.")

# Load Existing Profile Data
# ... (Keep loading logic) ...
existing_profile_dict = db_service.get_profile(user_id)
if existing_profile_dict is None: st.error("Could not load profile data."); existing_profile_dict = {}
elif not existing_profile_dict: st.caption("No profile found. Fill out the form.")
else: st.caption("Existing profile data loaded.")


# Define Profile Questions Options
# ... (Keep definitions) ...
age_ranges = ['18-24', '25-34', '35-44', '45-54', '55+']; income_ranges = ['< ₹5 LPA', '₹5-12 LPA', '₹12-25 LPA', '₹25+ LPA']
savings_levels = ['Low', 'Medium', 'High']; debt_levels = ['Low', 'Medium', 'High']; dependents_options = ['Yes', 'No']
goal_options = ['Retirement', 'ChildEdu', 'Property', 'Marriage', 'Business', 'Wealth', 'Other']
time_horizon_ranges = ['< 5 years', '5 - 10 years', '11 - 15 years', '16 - 20 years', 'More than 20 years']
time_horizon_map = {'< 5 years': 3, '5 - 10 years': 7, '11 - 15 years': 13, '16 - 20 years': 18, 'More than 20 years': 25}
tolerance_levels = ['Low', 'Medium', 'High']


# Helper functions
# ... (Keep helper functions) ...
def get_index(options_list, value):
    if value is None: return 0
    try: return options_list.index(value)
    except ValueError: print(f"Warning: Value '{value}' not in {options_list}."); return 0
def get_range_from_years(year_map, years_value):
    default_range = list(year_map.keys())[0]
    if years_value is None: return default_range
    for key, val in year_map.items():
        if val == years_value: return key
    print(f"Warning: Years value '{years_value}' not found."); return default_range


# Profile Form
with st.container():
    with st.form("profile_form"):
        # ... (Keep form layout) ...
        st.subheader("🧑 About You"); col1, col2 = st.columns(2)
        with col1: existing_age = existing_profile_dict.get("AgeRange"); age_range = st.selectbox("Age Range", age_ranges, index=get_index(age_ranges, existing_age))
        with col2: existing_dependents = existing_profile_dict.get("HasDependents"); has_dependents = st.radio("Financially Support Dependents?", dependents_options, index=get_index(dependents_options, existing_dependents), horizontal=True)
        st.markdown("---")
        st.subheader("💰 Financial Situation"); col1, col2 = st.columns(2)
        with col1:
            existing_income = existing_profile_dict.get("IncomeRange"); income_range = st.selectbox("Annual Household Income (INR)", income_ranges, index=get_index(income_ranges, existing_income))
            existing_savings = existing_profile_dict.get("SavingsLevel"); savings_level = st.selectbox("Savings Level", savings_levels, index=get_index(savings_levels, existing_savings), help="Compared to monthly essential expenses (Low: <3mo, Medium: 3-6mo, High: >6mo)")
        with col2:
            existing_debt = existing_profile_dict.get("DebtLevel"); debt_level = st.selectbox("Debt Level (Non-Mortgage)", debt_levels, index=get_index(debt_levels, existing_debt), help="Relative to income (Low: Minimal EMIs, Medium: Budgeted, High: Significant EMIs)")
        st.markdown("---")
        st.subheader("🎯 Goals & Timeline"); col1, col2 = st.columns(2)
        with col1: existing_goal = existing_profile_dict.get("PrimaryGoal"); primary_goal = st.selectbox("Primary Financial Goal", goal_options, index=get_index(goal_options, existing_goal))
        with col2: existing_years = existing_profile_dict.get("TimeHorizonYears"); time_horizon_range = st.select_slider("Time Horizon for Goal", options=time_horizon_ranges, value=get_range_from_years(time_horizon_map, existing_years))
        st.markdown("---")
        st.subheader("🎢 Investment Comfort")
        existing_tolerance = existing_profile_dict.get("SelfReportedTolerance"); self_reported_tolerance = st.radio("Investment Risk Tolerance", options=tolerance_levels, index=get_index(tolerance_levels, existing_tolerance), horizontal=True, help="Low: Prioritize safety, Medium: Balanced risk/reward, High: Seek higher growth with higher risk")
        st.markdown("---")

        submitted = st.form_submit_button("💾 Save / Update Profile")

# Handle Form Submission
if submitted:
    with st.spinner("⏳ Saving profile..."):
        time_horizon_years = time_horizon_map[time_horizon_range]
        profile_data = {"AgeRange": age_range, "IncomeRange": income_range, "SavingsLevel": savings_level, "DebtLevel": debt_level, "HasDependents": has_dependents, "PrimaryGoal": primary_goal, "TimeHorizonYears": time_horizon_years, "SelfReportedTolerance": self_reported_tolerance}
        try:
            saved_profile = db_service.save_or_update_profile(user_id, profile_data)
            if saved_profile:
                 st.success("✅ Profile saved successfully!")
                 time.sleep(1.5) # time.sleep is now valid
                 st.rerun()
            else: st.error("❌ Failed to save profile. Please check logs.")
        except Exception as e: st.error(f"❌ Failed to save profile: An error occurred."); print(f"Error saving profile: {e}"); traceback.print_exc()