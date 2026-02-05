import streamlit as st
from google_auth import handle_oauth_callback
from login_page import render_login_page
from ibm_app import run_ibm_app

st.set_page_config(page_title="Quantum State Visualizer", layout="wide")

# ---- OAuth callback FIRST ----
handle_google_callback()

# ---- Auth flags ----
if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = None

# ---- Hard router ----
if not st.session_state.is_authenticated:
    render_login_page()
    st.stop()

# ---- Protected app ----
run_ibm_app()
