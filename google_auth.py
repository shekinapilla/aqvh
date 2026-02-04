# google_auth.py
import streamlit as st
import os
from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
from google.auth.transport import requests
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/drive.file",
]

def _flow():
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": os.environ["GOOGLE_CLIENT_ID"],
                "client_secret": os.environ["GOOGLE_CLIENT_SECRET"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [os.environ["REDIRECT_URI"]],
            }
        },
        scopes=SCOPES,
    )
    flow.redirect_uri = os.environ["REDIRECT_URI"]
    return flow


def login_button():
    flow = _flow()

    # 🔴 IMPORTANT FIX: online access only
    auth_url, state = flow.authorization_url(
        access_type="online",
        prompt="select_account",
        include_granted_scopes="true",
    )

    st.session_state["oauth_state"] = state

    st.markdown(
        f"""
        <a href="{auth_url}" style="text-decoration:none;">
            <div class="google-login-btn">
                <img src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg">
                <span>Sign in with Google</span>
            </div>
        </a>
        """,
        unsafe_allow_html=True,
    )


def handle_callback():
    # 🔒 HARD STOP: never re-handle OAuth
    if st.session_state.get("google_logged_in"):
        return True

    qp = st.query_params
    if "code" not in qp:
        return False

    try:
        flow = _flow()
        flow.fetch_token(code=qp["code"])

        creds = flow.credentials

        idinfo = id_token.verify_oauth2_token(
            creds.id_token,
            requests.Request(),
            os.environ["GOOGLE_CLIENT_ID"],
        )

        st.session_state.google_logged_in = True
        st.session_state.google_email = idinfo["email"]
        st.session_state.google_creds = creds

        return True

    except Exception as e:
        st.error(f"Google authentication error: {e}")
        return False


def upload_history_to_drive(local_file):
    if not st.session_state.get("google_logged_in"):
        return

    creds = st.session_state["google_creds"]
    email = st.session_state["google_email"]

    drive = build("drive", "v3", credentials=creds)

    root_q = "name='QuantumVisualizer' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    res = drive.files().list(q=root_q, fields="files(id)").execute()
    files = res.get("files", [])

    if files:
        root_id = files[0]["id"]
    else:
        root = drive.files().create(
            body={"name": "QuantumVisualizer", "mimeType": "application/vnd.google-apps.folder"},
            fields="id",
        ).execute()
        root_id = root["id"]

    user_q = f"name='{email}' and '{root_id}' in parents and trashed=false"
    res = drive.files().list(q=user_q, fields="files(id)").execute()
    files = res.get("files", [])

    if files:
        folder_id = files[0]["id"]
    else:
        folder = drive.files().create(
            body={
                "name": email,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [root_id],
            },
            fields="id",
        ).execute()
        folder_id = folder["id"]

    media = MediaFileUpload(local_file, resumable=False)
    drive.files().create(
        body={"name": "history.pkl", "parents": [folder_id]},
        media_body=media,
    ).execute()
