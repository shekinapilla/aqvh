import os
import streamlit as st
from google_auth_oauthlib.flow import Flow

CLIENT_ID = os.environ["GOOGLE_CLIENT_ID"]
CLIENT_SECRET = os.environ["GOOGLE_CLIENT_SECRET"]
REDIRECT_URI = os.environ["REDIRECT_URI"]
SCOPES = ["openid", "email", "profile"]


def get_flow(state=None):
    return Flow.from_client_config(
        {
            "web": {
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [REDIRECT_URI],
            }
        },
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
        state=state,
    )


def start_google_login():
    flow = get_flow()
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    
    # ✅ Store state in session state (persistent across redirects)
    st.session_state.oauth_state = state
    
    # Store the auth URL in session too if needed
    st.session_state.auth_url = auth_url
    
    st.markdown(
        f'<meta http-equiv="refresh" content="0; url={auth_url}">',
        unsafe_allow_html=True,
    )


def handle_google_callback():
    params = st.query_params
    
    if "code" not in params or "state" not in params:
        return False
    
    # Get state from session state
    expected_state = st.session_state.get("oauth_state")
    returned_state = params["state"]
    
    if not expected_state:
        st.error("No OAuth state found in session. Please try logging in again.")
        return False
    
    if expected_state != returned_state:
        st.error("Invalid OAuth state. Possible CSRF attack or session expired.")
        # Clean up
        if "oauth_state" in st.session_state:
            del st.session_state.oauth_state
        return False
    
    try:
        flow = get_flow(state=returned_state)
        flow.fetch_token(code=params["code"])
        
        st.session_state.google_credentials = flow.credentials
        st.session_state.google_oauth_done = True
        
        # Clean up
        if "oauth_state" in st.session_state:
            del st.session_state.oauth_state
        
        # Clear query params
        st.query_params.clear()
        
        return True
        
    except Exception as e:
        st.error(f"Google authentication failed: {e}")
        return False
