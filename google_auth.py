import os
import streamlit as st
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials

CLIENT_ID = os.environ["GOOGLE_CLIENT_ID"]
CLIENT_SECRET = os.environ["GOOGLE_CLIENT_SECRET"]
REDIRECT_URI = os.environ["REDIRECT_URI"]

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

def get_oauth_flow(state=None):
    """Create OAuth flow instance"""
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

def initiate_google_login():
    """Start Google OAuth flow"""
    # Clear any existing state
    if "oauth_state" in st.session_state:
        del st.session_state.oauth_state
    
    flow = get_oauth_flow()
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    
    # Store state in session
    st.session_state.oauth_state = state
    st.session_state.oauth_flow = flow  # Store the entire flow if needed
    
    # Redirect
    st.markdown(
        f'<meta http-equiv="refresh" content="0; url={auth_url}">',
        unsafe_allow_html=True,
    )

def handle_oauth_callback():
    """Handle OAuth callback"""
    params = st.query_params
    
    # Check if this is a callback
    if "code" not in params or "state" not in params:
        return False
    
    # Verify state
    expected_state = st.session_state.get("oauth_state")
    returned_state = params.get("state")
    
    if not expected_state or expected_state != returned_state:
        st.error("Invalid or expired OAuth session. Please try again.")
        # Clean up
        for key in ["oauth_state", "oauth_flow"]:
            if key in st.session_state:
                del st.session_state[key]
        return False
    
    try:
        # Use the stored flow or create new one with state
        flow = st.session_state.get("oauth_flow") or get_oauth_flow(state=returned_state)
        
        # Exchange code for tokens
        flow.fetch_token(code=params["code"])
        
        # Store credentials
        creds = flow.credentials
        st.session_state.google_credentials = {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": creds.scopes,
            "id_token": creds.id_token,
            "expiry": creds.expiry.isoformat() if creds.expiry else None,
        }
        
        # Verify and extract user info from ID token
        if creds.id_token:
            from google.oauth2 import id_token
            from google.auth.transport import requests as google_requests
            
            try:
                idinfo = id_token.verify_oauth2_token(
                    creds.id_token,
                    google_requests.Request(),
                    CLIENT_ID,
                )
                st.session_state.google_email = idinfo.get("email")
                st.session_state.google_name = idinfo.get("name")
            except Exception as e:
                st.warning(f"Could not verify ID token: {e}")
        
        # Clean up
        for key in ["oauth_state", "oauth_flow"]:
            if key in st.session_state:
                del st.session_state[key]
        
        # Clear query params
        st.query_params.clear()
        
        st.success("Successfully logged in with Google!")
        return True
        
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        # Clean up on error
        for key in ["oauth_state", "oauth_flow"]:
            if key in st.session_state:
                del st.session_state[key]
        return False

def is_google_logged_in():
    """Check if user is logged in with Google"""
    return "google_credentials" in st.session_state

def get_google_creds():
    """Get Google credentials from session"""
    if "google_credentials" not in st.session_state:
        return None
    
    creds_data = st.session_state.google_credentials
    return Credentials(
        token=creds_data["token"],
        refresh_token=creds_data.get("refresh_token"),
        token_uri=creds_data["token_uri"],
        client_id=creds_data["client_id"],
        client_secret=creds_data["client_secret"],
        scopes=creds_data["scopes"],
        id_token=creds_data.get("id_token"),
    )
