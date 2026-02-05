import streamlit as st
from google_auth import handle_oauth_callback
from login_page import render_login_page
from ibm_app import run_ibm_app

st.set_page_config(page_title="Quantum State Visualizer", layout="wide")

# ---- OAuth callback FIRST ----
# Idi Google login response process chesthundi
if handle_oauth_callback():  # Login successful ayithe
    st.session_state.is_authenticated = True
    st.session_state.auth_mode = "google"
    # Immediately app open cheyadaniki
    st.rerun()

# ---- Auth flags ----
# Initialize auth state
if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = None

# ---- Google credentials check ----
# Google login ayi credentials unte, automatically authenticate cheyyali
if "google_credentials" in st.session_state:
    st.session_state.is_authenticated = True
    st.session_state.auth_mode = "google"

# ---- Hard router ----
# Login avvaledhu ante login page chupiyali
if not st.session_state.is_authenticated:
    render_login_page()
    st.stop()

# ---- Protected app ----
# Login ayithe direct app open avvali
run_ibm_app()
