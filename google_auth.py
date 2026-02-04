import streamlit as st
import os
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.oauth2 import id_token
from google.auth.transport import requests
from googleapiclient.http import MediaFileUpload
import time

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
    """Handle Google OAuth callback - SIMPLIFIED AND FIXED"""
    query_params = st.query_params.to_dict()
    
    # Exit if no code
    if "code" not in query_params:
        return

    # Get the authorization code
    incoming_code = query_params["code"]
    
    # Check if we already processed this code
    code_key = f"processed_code_{incoming_code}"
    if st.session_state.get(code_key):
        return  # Already processed
    
    # Mark as processed
    st.session_state[code_key] = True
    
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
        
        # Get state from session or query params
        state = query_params.get("state") or st.session_state.get("oauth_state")
        
        # Fetch token
        flow.fetch_token(code=incoming_code, state=state)
        
        creds = flow.credentials
        
        # Verify the ID token
        idinfo = id_token.verify_oauth2_token(
            creds.id_token,
            requests.Request(),
            os.environ["GOOGLE_CLIENT_ID"],
        )

        # Update session state
        st.session_state["google_email"] = idinfo["email"]
        st.session_state["google_creds"] = creds
        st.session_state["google_logged_in"] = True
        
        # Success
        return True
        
    except Exception as e:
        st.error(f"Google authentication error: {str(e)}")
        # Remove the processed flag so user can try again
        st.session_state.pop(code_key, None)
        return False

def get_drive_service():
    creds = st.session_state["google_creds"]
    return build("drive", "v3", credentials=creds)


def get_or_create_folder(service, name, parent_id=None):
    q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        q += f" and '{parent_id}' in parents"

    results = service.files().list(q=q, fields="files(id)").execute()
    files = results.get("files", [])

    if files:
        return files[0]["id"]

    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    if parent_id:
        metadata["parents"] = [parent_id]

    folder = service.files().create(body=metadata, fields="id").execute()
    return folder["id"]

def upload_history_to_drive(local_file="history.pkl"):
    if "google_creds" not in st.session_state:
        st.warning("❌ No Google credentials")
        return

    if "google_email" not in st.session_state:
        st.warning("❌ No Google email")
        return

    if not os.path.exists(local_file):
        st.error(f"❌ File not found: {local_file}")
        return

    creds = st.session_state["google_creds"]
    email = st.session_state["google_email"]

    drive = build("drive", "v3", credentials=creds)

    # 1️⃣ App root folder
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

    # 2️⃣ User email folder
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

    # 3️⃣ Upload file
    media = MediaFileUpload(local_file, resumable=False)

    drive.files().create(
        body={"name": "history.pkl", "parents": [folder_id]},
        media_body=media,
        fields="id",
    ).execute()

    st.success("✅ history.pkl uploaded to Google Drive")
