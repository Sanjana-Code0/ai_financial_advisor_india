# streamlit_app/pages/3_📊_Dashboard_Advice.py
# --- Add sys.path block at the VERY TOP ---
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of block ---

import streamlit as st
try:
    from services import advice_service, db_service
    from utils import load_css # Import CSS loader
except ImportError as e:
    st.error(f"Failed to import modules: {e}. Ensure you run from project root and venv is active.")
    st.stop()
import matplotlib.pyplot as plt # For pie charts
import pandas as pd
import numpy as np # For pie chart data if returns are negative
import time # For st.success message display timing

# --- Load CSS ---
load_css("style.css") # Make sure style.css is in streamlit_app/

st.header("📊 Dashboard & Financial Advice")
st.write("Your personalized financial snapshot and guidance.")
st.markdown("---")

# --- Calculator Functions (defined at the top for clarity) ---
def calculate_sip_investment(monthly_investment, annual_return_rate, time_period_years):
    if annual_return_rate < -99: annual_return_rate = -99 # Cap extreme negatives
    if time_period_years <= 0 or monthly_investment < 0: return 0, 0, 0 # monthly_investment can be 0
    monthly_rate = (annual_return_rate / 100) / 12
    number_of_months = time_period_years * 12
    invested_amount = monthly_investment * number_of_months
    future_value = invested_amount # Start with invested amount for 0% or negative adjusted rates
    try:
        if monthly_rate > (-1/number_of_months if number_of_months > 0 else 0): # Avoid issues with very negative rates
            if monthly_rate == 0: # Handles 0% return rate
                future_value = invested_amount
            else:
                future_value = monthly_investment * (((1 + monthly_rate)**number_of_months - 1) / monthly_rate)
        else: # For very negative rates, FV can be less than 0, cap at 0 for practicality
            future_value = 0
        if future_value < 0: future_value = 0 # Final cap
        estimated_returns = future_value - invested_amount
        return round(invested_amount), round(estimated_returns), round(future_value)
    except (OverflowError, ValueError): # Handle math errors
        return invested_amount, -invested_amount if invested_amount > 0 else 0, 0


def calculate_lumpsum_investment(principal_amount, annual_return_rate, time_period_years):
    if annual_return_rate < -99: annual_return_rate = -99 # Cap extreme negatives
    if time_period_years <= 0 or principal_amount < 0: return principal_amount, 0, principal_amount # principal can be 0
    invested_amount = principal_amount
    future_value = invested_amount # Start with invested amount
    try:
        # Compound interest formula: A = P(1 + r)^t
        if (1 + (annual_return_rate / 100)) >= 0: # Avoid issues with negative base in power
            future_value = principal_amount * ((1 + (annual_return_rate / 100)) ** time_period_years)
        else: # If rate implies losing more than 100% annually, cap future value at 0
            future_value = 0
        if future_value < 0: future_value = 0 # Final cap
        estimated_returns = future_value - invested_amount
        return round(invested_amount), round(estimated_returns), round(future_value)
    except (OverflowError, ValueError):
        return invested_amount, -invested_amount if invested_amount > 0 else 0, 0


# --- Login & Profile Checks ---
if not st.session_state.get('logged_in'):
    st.warning("Please login first to access the dashboard.")
    st.stop()
user_id = st.session_state.get('user_id')
if not user_id:
    st.error("Error: User ID not found in session. Please login again.")
    st.stop()
profile_complete = db_service.is_profile_complete(user_id)
if not profile_complete:
    st.warning("Please complete your profile on the '👤 Profile' page to get advice.")
    st.stop()

st.info(f"👋 Welcome, **{st.session_state.get('username', 'User')}**!")

# --- Inputs for Investment Recommendation Projections (in sidebar) ---
st.sidebar.subheader("Investment Projection Settings")
projection_amount_for_recs = st.sidebar.number_input(
    "Hypothetical Amount for Rec. Projections (₹)",
    min_value=1000, max_value=10000000, value=50000, step=1000, key="proj_amt_recs",
    help="This amount will be used to show potential growth for suitable investments after you click 'Get My Financial Advice'."
)
projection_years_for_recs = st.sidebar.slider(
    "Projection Horizon for Recs. (Years)",
    min_value=1, max_value=30, value=5, step=1, key="proj_yrs_recs",
    help="How many years into the future to project growth for recommended investments."
)
st.sidebar.markdown("---") # Divider in sidebar

# --- "Get My Financial Advice" Section ---
if st.button("🚀 Get My Financial Advice"):
    st.markdown("---")
    with st.spinner("⏳ Analyzing profile & generating advice..."):
        advice_result = advice_service.generate_advice(
            user_id,
            projection_principal_ui=projection_amount_for_recs, # Pass sidebar values
            projection_years_ui=projection_years_for_recs
        )

        if advice_result and "error" not in advice_result:
            # --- Display Financial Assessment ---
            with st.container(): # Card-like container for assessment
                st.subheader("📈 Your Financial Assessment")
                col1, col2 = st.columns(2)
                risk_profile = advice_result.get("risk_profile", "N/A")
                with col1: st.metric(label="Predicted Risk Profile", value=risk_profile)
                with col2:
                     profile_info = db_service.get_profile(user_id)
                     st.metric(label="Primary Goal", value=profile_info.get("PrimaryGoal", "N/A") if profile_info else "N/A")

                st.markdown("**Understanding Your Risk Profile:**")
                # Display Static Explanation First
                st.write(advice_result.get("risk_explanation_simple", "*Explanation unavailable.*"))
                st.markdown("---") # Separator
                # Display Detailed SHAP Explanation in an Expander
                with st.expander("View Detailed Factors (AI Analysis - Why this profile?)", expanded=False): # Default to collapsed
                    st.markdown(advice_result.get("risk_explanation_detailed_shap", "*Detailed factor analysis unavailable.*"))

            st.markdown("<br>", unsafe_allow_html=True) # Add some vertical space

            # --- Display Investment Recommendations with Projections ---
            with st.container(): # Card-like container for investments
                st.subheader("💡 Investment Recommendations")
                investment_recs = advice_result.get("investment_recommendations", [])
                if not investment_recs: st.info("*No specific investment recommendations generated.*")
                elif investment_recs[0].get("investment") == "Error": st.warning(f"*Could not generate recommendations: {investment_recs[0].get('explanation')}*")
                elif investment_recs[0].get("investment") == "None Suitable": st.info(investment_recs[0].get("explanation"))
                else:
                    st.write(f"**Based on your risk profile, these approaches might be suitable. Projections below are for an illustrative initial investment of ₹{projection_amount_for_recs:,.0f} over {projection_years_for_recs} years:**")
                    num_cols = 2; cols = st.columns(num_cols); col_idx = 0
                    for rec in investment_recs:
                        with cols[col_idx % num_cols]:
                            st.markdown(f"##### **{rec.get('investment', 'Unknown')}**")
                            if rec.get("projected_value", 0) > 0 and rec.get("avg_annual_return_used") != "N/A":
                                # Ensure avg_annual_return_used is float for formatting
                                try: avg_return_display = f"{float(rec.get('avg_annual_return_used', 0.0)):.1f}"
                                except ValueError: avg_return_display = "N/A"
                                st.markdown(f"Est. Growth (avg. {avg_return_display}% p.a.):")
                                proj_col1, proj_col2 = st.columns(2)
                                with proj_col1: st.metric(label="Projected Value", value=f"₹{rec.get('projected_value', 0):,.0f}")
                                with proj_col2: st.metric(label="Total Growth", value=f"₹{rec.get('total_growth', 0):,.0f}")
                            else: st.caption("*Projection not shown.*")
                            with st.expander("View Rationale (AI Analysis)", expanded=False):
                                st.markdown(rec.get('explanation', '*Detailed rationale unavailable.*'))
                        col_idx += 1
                        if col_idx % num_cols == 0 and col_idx < len(investment_recs): # Add divider between full rows
                            st.markdown("<hr style='margin-top:0.1em; margin-bottom:0.1em; border:0; border-top: 1px solid #eee;'/>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # --- Display Personalized Planning Actions (Placeholder) ---
            with st.container(): # Card-like
                st.subheader("🧭 Personalized Planning Actions (Placeholder)")
                planning_rec = advice_result.get("planning_recommendation")
                if planning_rec:
                     with st.expander("View Suggested Actions & Rationale", expanded=False):
                         st.markdown("**Suggested Actions:**")
                         action_list = planning_rec.get("actions", ["*Not available*"])
                         if action_list and action_list[0] != "*Not available*":
                             for item in action_list:
                                 if isinstance(item, str): st.markdown(f"- {item}")
                                 else: st.markdown(f"- *Could not display action: {str(item)}*")
                         else: st.markdown("*No specific actions suggested.*")
                         st.markdown("\n**Rationale:**")
                         st.markdown(planning_rec.get("explanation", "*Explanation not available.*"))
                else: st.info("*Planning recommendations are currently unavailable.*")

            st.markdown("---"); st.success("✅ Advice generated successfully!")
        elif advice_result and "error" in advice_result: st.error(f"❌ Could not generate advice: {advice_result['error']}")
        else: st.error("❌ An unexpected error occurred.")
else: # This else corresponds to: if st.button("🚀 Get My Financial Advice")
    st.markdown("Set projection inputs in the sidebar and click the '🚀 Get My Financial Advice' button above for your assessment.")
# --- END OF ADVICE GENERATION BLOCK ---

st.markdown("---") # This divider should be at the base level

# --- Investment Growth Calculators Section ---
st.subheader("💰 Investment Growth Calculators")
st.write("Estimate the potential growth of your investments. These are illustrative and not guaranteed.")

# Use session state for better interactivity between number_input and slider
if 'sip_monthly_inv_calc' not in st.session_state: st.session_state.sip_monthly_inv_calc = 10000
if 'sip_return_calc' not in st.session_state: st.session_state.sip_return_calc = 12.0
if 'sip_years_calc' not in st.session_state: st.session_state.sip_years_calc = 10

if 'lump_principal_calc' not in st.session_state: st.session_state.lump_principal_calc = 100000
if 'lump_return_calc' not in st.session_state: st.session_state.lump_return_calc = 12.0
if 'lump_years_calc' not in st.session_state: st.session_state.lump_years_calc = 10


def create_donut_chart(invested, returns, colors=['#3B82F6', '#A7C7E7']): # Blue shades
    if invested <= 0 and returns <= 0: # Avoid plotting empty chart
        return None
    labels = ['Invested', 'Est. Returns']
    # Handle negative returns for pie chart visualization (show as zero proportion if negative)
    sizes = [invested, max(0, returns)]
    if invested <= 0 and returns > 0: # Only returns, no principal
        labels = ['Est. Returns']
        sizes = [returns]
        colors = [colors[1]]
    elif returns <= 0 and invested > 0: # Only principal, no gains or loss shown in returns part
        labels = ['Invested']
        sizes = [invested]
        colors = [colors[0]]


    fig, ax = plt.subplots(figsize=(3.5, 3.5)) # Slightly larger for clarity
    ax.pie(sizes, labels=None, autopct=None, startangle=90, colors=colors,
           wedgeprops=dict(width=0.4, edgecolor='white')) # width creates the donut hole
    centre_circle = plt.Circle((0,0),0.60,fc='white')
    fig.gca().add_artist(centre_circle)
    ax.axis('equal')
    plt.tight_layout(pad=0.1)
    return fig


calc_tab1, calc_tab2 = st.tabs(["**SIP Calculator**", "**Lumpsum Calculator**"])

with calc_tab1:
    st.markdown("##### Systematic Investment Plan (SIP)")
    c1, c2 = st.columns([0.6, 0.4]) # Adjust column ratio
    with c1: # Inputs and Results
        st.session_state.sip_monthly_inv_calc = st.number_input("Monthly Investment (₹)", min_value=0, max_value=200000,value=st.session_state.sip_monthly_inv_calc, step=500, key="sip_m_num")
        st.session_state.sip_return_calc = st.slider("Expected Annual Return Rate (%)", 0.0, 30.0, st.session_state.sip_return_calc, 0.5, key="sip_r_slider", format="%.1f%%")
        st.session_state.sip_years_calc = st.slider("Investment Duration (Years)", 1, 40, st.session_state.sip_years_calc, 1, key="sip_y_slider")

        sip_invested, sip_returns, sip_total_value = calculate_sip_investment(st.session_state.sip_monthly_inv_calc, st.session_state.sip_return_calc, st.session_state.sip_years_calc)

        st.markdown("<hr style='margin:0.5em 0; border-top: 1px solid #eee;'/>", unsafe_allow_html=True)
        res_col1, res_col2 = st.columns(2)
        with res_col1: st.markdown(f"**Invested Amount:**<br>₹{sip_invested:,.0f}", unsafe_allow_html=True)
        with res_col2: st.markdown(f"**Est. Returns:**<br>₹{sip_returns:,.0f}", unsafe_allow_html=True)
        st.markdown(f"#### **Projected Value:** <span style='color:#0a3d62;'>₹{sip_total_value:,.0f}</span>", unsafe_allow_html=True)

    with c2: # Chart
        st.markdown("<p style='text-align:center; font-weight:bold; margin-bottom:0px;'>Value Distribution</p>", unsafe_allow_html=True)
        fig_sip = create_donut_chart(sip_invested, sip_returns)
        if fig_sip: st.pyplot(fig_sip, use_container_width=True)
        else: st.caption("Enter values to see chart.")


with calc_tab2:
    st.markdown("##### Lumpsum Investment")
    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        st.session_state.lump_principal_calc = st.number_input("Total Investment (₹)", min_value=500, max_value=20000000, value=st.session_state.lump_principal_calc, step=1000, key="lump_p_num")
        st.session_state.lump_return_calc = st.slider("Expected Annual Return Rate (%)", 0.0, 30.0, st.session_state.lump_return_calc, 0.5, key="lump_r_slider", format="%.1f%%")
        st.session_state.lump_years_calc = st.slider("Investment Duration (Years)", 1, 40, st.session_state.lump_years_calc, 1, key="lump_y_slider")

        lump_invested, lump_returns, lump_total_value = calculate_lumpsum_investment(st.session_state.lump_principal_calc, st.session_state.lump_return_calc, st.session_state.lump_years_calc)

        st.markdown("<hr style='margin:0.5em 0; border-top: 1px solid #eee;'/>", unsafe_allow_html=True)
        res_col1, res_col2 = st.columns(2)
        with res_col1: st.markdown(f"**Principal Amount:**<br>₹{lump_invested:,.0f}", unsafe_allow_html=True)
        with res_col2: st.markdown(f"**Est. Returns:**<br>₹{lump_returns:,.0f}", unsafe_allow_html=True)
        st.markdown(f"#### **Projected Value:** <span style='color:#0a3d62;'>₹{lump_total_value:,.0f}</span>", unsafe_allow_html=True)

    with c2: # Chart
        st.markdown("<p style='text-align:center; font-weight:bold; margin-bottom:0px;'>Value Distribution</p>", unsafe_allow_html=True)
        fig_lump = create_donut_chart(lump_invested, lump_returns, colors=['#ff9966', '#ffcc66']) # Different colors
        if fig_lump: st.pyplot(fig_lump, use_container_width=True)
        else: st.caption("Enter values to see chart.")

st.caption("Note: These calculators provide estimations based on expected returns and do not guarantee actual returns. Market risks apply.")
# --- End of Calculator Section ---

st.markdown("---");
st.caption("Disclaimer: Projections are illustrative and not guaranteed. Consult a professional financial advisor.")