import os
from dotenv import load_dotenv
import streamlit as st
from google_auth_oauthlib.flow import Flow

# 🔥 LOAD ENV FIRST
load_dotenv()

CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.environ.get("REDIRECT_URI")
if not CLIENT_ID or not CLIENT_SECRET or not REDIRECT_URI:
    raise RuntimeError(
        "Missing Google OAuth env vars. "
        "Check GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, REDIRECT_URI"
    )


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

    # 🔒 Private session key (critical)
    st.session_state["_oauth_state"] = state
    st.session_state["_oauth_started"] = True

    st.markdown(
        f'<meta http-equiv="refresh" content="0; url={auth_url}">',
        unsafe_allow_html=True,
    )


def handle_google_callback():
    # Already authenticated → do nothing
    if st.session_state.get("google_oauth_done"):
        return True

    params = st.query_params

    if "code" not in params or "state" not in params:
        return False

    # 🔒 Lock state once
    if "_oauth_state_checked" not in st.session_state:
        expected = st.session_state.get("_oauth_state")
        returned = params["state"][0]

        if not expected or returned != expected:
            st.error("Invalid OAuth state")
            return False

        st.session_state["_oauth_state_checked"] = True

    try:
        flow = get_flow(state=params["state"][0])
        flow.fetch_token(code=params["code"][0])

        st.session_state.google_credentials = flow.credentials
        st.session_state.google_oauth_done = True

        # Cleanup AFTER success
        st.session_state.pop("_oauth_state", None)
        st.session_state.pop("_oauth_state_checked", None)
        st.query_params.clear()

        return True

    except Exception as e:
        st.error(f"Google authentication failed: {e}")
        return False

