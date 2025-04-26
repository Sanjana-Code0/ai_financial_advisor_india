# --- Add this block at the VERY TOP ---
import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path: sys.path.insert(0, project_root)
# --- End of block ---

import streamlit as st
import time # <<<<<<<<<<< ADD THIS IMPORT
try: from services import auth_service, db_service
except ImportError as e: st.error(f"Failed to import services (Login Page): {e}."); st.stop()
import traceback

# NO st.set_page_config() here
st.header("🔐 Login or Register")
st.write("Access your personalized financial insights.")
st.markdown("---")

# Check if already logged in
if st.session_state.get('logged_in'):
    st.success(f"You are already logged in as {st.session_state.get('username')}.")
    if st.button("Logout", key="login_page_logout"):
        st.session_state.logged_in = False; st.session_state.user_id = None; st.session_state.username = None
        st.success("You have been logged out."); time.sleep(1); st.rerun() # Use st.rerun()
    st.stop()

# Use tabs for Login and Registration forms
login_tab, register_tab = st.tabs(["**Login**", "**Register**"])

with login_tab:
    with st.container():
        st.subheader("Login to Your Account")
        with st.form("login_form"):
            login_username = st.text_input("Username", key="login_user")
            login_password = st.text_input("Password", type="password", key="login_pass")
            login_submitted = st.form_submit_button("Login")

            if login_submitted:
                if not login_username or not login_password: st.warning("⚠️ Please enter both username and password.")
                else:
                    user_auth_info = auth_service.authenticate_user(login_username, login_password)
                    if user_auth_info:
                        st.session_state.logged_in = True; st.session_state.user_id = user_auth_info["id"]; st.session_state.username = user_auth_info["username"]
                        st.success("✅ Login Successful!"); time.sleep(1); st.rerun() # time.sleep is now valid
                    else:
                        st.error("❌ Invalid username or password.")
                        st.session_state.logged_in = False; st.session_state.user_id = None; st.session_state.username = None

with register_tab:
     # ... (Rest of registration code remains the same) ...
    with st.container():
        st.subheader("Create a New Account")
        with st.form("register_form"):
            reg_username = st.text_input("Choose a Username", key="reg_user")
            reg_password = st.text_input("Choose a Password", type="password", key="reg_pass", help="Minimum 6 characters")
            reg_password_confirm = st.text_input("Confirm Password", type="password", key="reg_pass_confirm")
            register_submitted = st.form_submit_button("Register")

            if register_submitted:
                if not reg_username or not reg_password or not reg_password_confirm: st.warning("⚠️ Please fill in all fields.")
                elif reg_password != reg_password_confirm: st.error("❌ Passwords do not match.")
                elif len(reg_password) < 6: st.warning("⚠️ Password should be at least 6 characters long.")
                else:
                    existing_user_data = db_service.get_user_auth_data_by_username(reg_username)
                    if existing_user_data: st.error("❌ Username already exists. Please choose another.")
                    else:
                        hashed_password = auth_service.get_password_hash(reg_password)
                        try:
                            user_id = db_service.create_user(reg_username, hashed_password)
                            if user_id: st.success(f"✅ Registration successful for '{reg_username}'! Please login using the Login tab.")
                            else: st.error(f"❌ Registration failed: Could not create user.")
                        except Exception as e:
                            st.error(f"❌ Registration failed: An error occurred."); print(f"Reg Exception: {e}"); traceback.print_exc()