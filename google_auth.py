import os
import streamlit as st
from google_auth_oauthlib.flow import Flow

CLIENT_ID = os.environ["GOOGLE_CLIENT_ID"]
CLIENT_SECRET = os.environ["GOOGLE_CLIENT_SECRET"]
REDIRECT_URI = os.environ["REDIRECT_URI"]

SCOPES = ["openid", "email", "profile"]


def create_flow():
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
    )


def start_google_login():
    flow = create_flow()
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )

    st.markdown(
        f'<meta http-equiv="refresh" content="0; url={auth_url}">',
        unsafe_allow_html=True,
    )


def handle_google_callback():
    if "code" not in st.query_params:
        return False

    try:
        flow = create_flow()
        flow.fetch_token(code=st.query_params["code"])

        st.session_state.google_credentials = flow.credentials
        st.session_state.google_oauth_done = True

        st.query_params.clear()
        return True

    except Exception as e:
        st.error(f"Google login failed: {e}")
        return False
