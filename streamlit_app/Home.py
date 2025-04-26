# streamlit_app/Home.py
import streamlit as st
from utils import load_css # Import the utility function
# Import the init function - Ensure db_service is importable
try:
    from services.db_service import init_db
except ImportError:
     st.error("Failed to import services. Check structure/run command.")
     st.stop()

# --- Page Configuration (ONLY HERE) ---
st.set_page_config(
    page_title="FinSight AI Advisor", # Updated Title
    page_icon="💡", # Updated Icon
    layout="wide",
    initial_sidebar_state="expanded" # Keep sidebar open initially
)

# --- Load CSS ---
load_css("style.css") # Load custom styles

# --- Initialize Database ---
try: init_db()
except Exception as e: st.error(f"Database initialization failed: {e}"); st.stop()

# --- Session State Initialization ---
st.session_state.setdefault('logged_in', False)
st.session_state.setdefault('user_id', None)
st.session_state.setdefault('username', None)

# --- Page Content ---
st.title("💡 FinSight AI Advisor")
st.caption("Transparent Financial Guidance Tailored for India")
st.markdown("---")

# --- Use columns for a better intro layout ---
col1, col2 = st.columns([2, 1]) # Give more space to text

with col1:
    if st.session_state.logged_in:
        st.success(f"👋 Welcome back, **{st.session_state.username}**!")
        st.markdown("""
        Navigate using the sidebar to:
        *   **👤 Profile:** Update your financial details.
        *   **📊 Dashboard & Advice:** Get your personalized assessment.
        """)
        if st.button("Logout", key="home_logout"):
            st.session_state.logged_in = False; st.session_state.user_id = None; st.session_state.username = None
            st.success("You have been logged out."); time.sleep(1); st.rerun()
    else:
        st.info("🔑 Please **Login** or **Register** using the sidebar to begin.")
        st.markdown("""
        **FinSight AI helps you understand your financial standing and potential investment paths with clear, explainable insights.**
        """)

    st.markdown("""
    ### How It Works
    1.  **Secure Access:** Register or Login.
    2.  **Build Your Profile:** Answer simple questions about your finances & goals.
    3.  **Get Insights:** Receive AI-driven risk assessment & suitable investment ideas.
    4.  **Understand Why:** Get clear explanations for the recommendations.
    """)

with col2:
    # Placeholder for an image or graphic later
    st.image("https://cdn-icons-png.flaticon.com/512/9088/9088785.png", width=200) # Example icon
    st.markdown("<br>", unsafe_allow_html=True) # Add some space
    if not st.session_state.logged_in:
         st.markdown("_(Navigate to 'Login / Register' in the sidebar)_")


st.markdown("---")
st.caption("Disclaimer: This tool provides suggestions, not professional financial advice. Consult a qualified advisor for decisions.")