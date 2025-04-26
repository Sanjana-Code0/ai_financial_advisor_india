# streamlit_app/utils.py
import streamlit as st
import os

def load_css(file_name):
    """Loads a CSS file and injects it into the Streamlit app."""
    css_path = os.path.join(os.path.dirname(__file__), file_name)
    try:
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        print(f"CSS loaded successfully from {file_name}")
    except FileNotFoundError:
        print(f"Warning: CSS file not found at {css_path}")
    except Exception as e:
        print(f"Error loading CSS: {e}")