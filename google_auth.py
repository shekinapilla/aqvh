import streamlit as st
import os
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.oauth2 import id_token
from google.auth.transport import requests
from googleapiclient.http import MediaFileUpload
import json

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/drive.file",
]

def login_button():
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
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    
    # Store state in session
    st.session_state["oauth_state"] = state
    
    st.markdown(f"""
    <a href="{auth_url}" style="text-decoration:none;">
        <div class="google-login-btn">
            <img src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg">
            <span>Sign in with Google</span>
        </div>
    </a>
    """, unsafe_allow_html=True)

def handle_callback():
    # Only process if we have code AND not already logged in
    if "code" not in st.query_params:
        return
        
    # Check if we already processed this login
    if st.session_state.get("login_processed"):
        st.query_params.clear()
        return
        
    try:
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
        
        # Get state from session
        state = st.session_state.get("oauth_state", "")
        
        flow.fetch_token(
            code=st.query_params["code"],
            state=state
        )

        creds = flow.credentials

        # Verify ID token
        idinfo = id_token.verify_oauth2_token(
            creds.id_token,
            requests.Request(),
            os.environ["GOOGLE_CLIENT_ID"],
        )

        # Store credentials
        st.session_state["google_email"] = idinfo["email"]
        st.session_state["google_creds"] = {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": creds.scopes,
            "id_token": creds.id_token
        }
        st.session_state["google_logged_in"] = True
        st.session_state["auth_mode"] = "google"
        st.session_state["login_processed"] = True
        
        # Store login info in query params temporarily
        st.query_params.clear()
        st.query_params["logged_in"] = "true"
        st.query_params["auth_mode"] = "google"
        st.query_params["email"] = idinfo["email"]
        
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        # Clear invalid state
        for key in ["oauth_state", "login_processed"]:
            if key in st.session_state:
                del st.session_state[key]

def check_persistent_login():
    """Check for persistent login via query params or session"""
    # Check if we have login info in query params
    if "logged_in" in st.query_params:
        auth_mode = st.query_params.get("auth_mode")
        email = st.query_params.get("email")
        
        if auth_mode == "google" and email:
            st.session_state.auth_mode = "google"
            st.session_state.google_email = email
            st.session_state.google_logged_in = True
            
            # Clear the query params
            st.query_params.clear()
            
        elif auth_mode == "local" and email:
            st.session_state.auth_mode = "local"
            st.session_state.local_email = email
            
            # Clear the query params
            st.query_params.clear()

def get_drive_service():
    if "google_creds" not in st.session_state:
        return None
    
    creds_data = st.session_state["google_creds"]
    creds = Credentials(
        token=creds_data["token"],
        refresh_token=creds_data["refresh_token"],
        token_uri=creds_data["token_uri"],
        client_id=creds_data["client_id"],
        client_secret=creds_data["client_secret"],
        scopes=creds_data["scopes"]
    )
    return build("drive", "v3", credentials=creds)

def upload_history_to_drive(local_file="history.pkl"):
    if "google_creds" not in st.session_state:
        return False

    if not os.path.exists(local_file):
        return False

    try:
        creds_data = st.session_state["google_creds"]
        email = st.session_state["google_email"]

        creds = Credentials(
            token=creds_data["token"],
            refresh_token=creds_data["refresh_token"],
            token_uri=creds_data["token_uri"],
            client_id=creds_data["client_id"],
            client_secret=creds_data["client_secret"],
            scopes=creds_data["scopes"]
        )
        
        drive = build("drive", "v3", credentials=creds)

        # Create root folder
        root_q = "name='QuantumVisualizer' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        root_res = drive.files().list(q=root_q, fields="files(id)").execute()
        root_files = root_res.get("files", [])

        if root_files:
            root_id = root_files[0]["id"]
        else:
            root = drive.files().create(
                body={"name": "QuantumVisualizer", "mimeType": "application/vnd.google-apps.folder"},
                fields="id"
            ).execute()
            root_id = root["id"]

        # Create user folder
        user_q = f"name='{email}' and mimeType='application/vnd.google-apps.folder' and '{root_id}' in parents and trashed=false"
        user_res = drive.files().list(q=user_q, fields="files(id)").execute()
        user_files = user_res.get("files", [])

        if user_files:
            folder_id = user_files[0]["id"]
        else:
            folder = drive.files().create(
                body={
                    "name": email,
                    "mimeType": "application/vnd.google-apps.folder",
                    "parents": [root_id],
                },
                fields="id"
            ).execute()
            folder_id = folder["id"]

        # Upload file
        media = MediaFileUpload(local_file, resumable=False)
        
        drive.files().create(
            body={"name": "history.pkl", "parents": [folder_id]},
            media_body=media,
            fields="id",
        ).execute()
        
        return True
        
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return False
