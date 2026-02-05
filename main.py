import streamlit as st
from google_auth import handle_oauth_callback  # This is what you're importing
from login_page import render_login_page
from ibm_app import run_ibm_app

st.set_page_config(page_title="Quantum State Visualizer", layout="wide")

# ---- OAuth callback FIRST ----
# Use the function you actually imported
if handle_oauth_callback():  # Changed from handle_google_callback()
    # If login successful, set session flags and rerun
    st.session_state.is_authenticated = True
    st.session_state.auth_mode = "google"
    st.rerun()

# ---- Auth flags ----
if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = None

# Check if Google credentials exist (for persistence)
if "google_credentials" in st.session_state and not st.session_state.is_authenticated:
    st.session_state.is_authenticated = True
    st.session_state.auth_mode = "google"

# ---- Hard router ----
if not st.session_state.is_authenticated:
    render_login_page()
    st.stop()

# ---- Protected app ----
run_ibm_app()
