import streamlit as st
import os
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.oauth2 import id_token
from google.auth.transport import requests

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
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )

    st.markdown(f"### 🔐 [Login with Google]({auth_url})")

def handle_callback():
    if st.session_state.get("google_logged_in"):
        return

    query_params = st.query_params
    if "code" not in query_params:
        return

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
    flow.fetch_token(code=query_params["code"])

    creds = flow.credentials

    # ✅ Decode ID token to get email
    idinfo = id_token.verify_oauth2_token(
        creds.id_token,
        requests.Request(),
        os.environ["GOOGLE_CLIENT_ID"],
    )

    st.session_state["google_email"] = idinfo["email"]
    st.session_state["google_creds"] = creds
    st.session_state["google_logged_in"] = True

    st.query_params.clear()
    
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
    """
    Upload history.pkl to Google Drive under a folder named by user email
    """
    if "google_creds" not in st.session_state:
        return

    if "user_email" not in st.session_state:
        return

    creds = st.session_state["google_creds"]
    email = st.session_state["user_email"]

    drive = build("drive", "v3", credentials=creds)

    # 1️⃣ Check if folder already exists
    query = f"name='{email}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = drive.files().list(q=query, fields="files(id)").execute()
    files = results.get("files", [])

    if files:
        folder_id = files[0]["id"]
    else:
        # 2️⃣ Create folder
        folder_metadata = {
            "name": email,
            "mimeType": "application/vnd.google-apps.folder",
        }
        folder = drive.files().create(body=folder_metadata, fields="id").execute()
        folder_id = folder["id"]

    # 3️⃣ Upload / overwrite history.pkl
    from googleapiclient.http import MediaFileUpload

    file_metadata = {
        "name": "history.pkl",
        "parents": [folder_id],
    }

    media = MediaFileUpload(local_file, resumable=True)

    drive.files().create(
        body=file_metadata,
        media_body=media,
        fields="id",
    ).execute()
