import streamlit as st
import os
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.oauth2 import id_token
from google.auth.transport import requests
from googleapiclient.http import MediaFileUpload

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

    st.markdown(f"""
    <a href="{auth_url}" style="text-decoration:none;">
        <div class="google-login-btn">
            <img src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg">
            <span>Sign in with Google</span>
        </div>
    </a>
    """, unsafe_allow_html=True)

def handle_callback():
    """Handle Google OAuth callback with proper error handling"""
    query_params = st.query_params
    
    # 1. Exit immediately if there's no 'code' parameter
    if "code" not in query_params:
        return

    # 2. Get the authorization code
    incoming_code = query_params["code"]
    
    # 3. Create a unique lock key for this authorization code
    lock_key = f"_oauth_handled_for_code_{incoming_code}"
    
    # 4. If this specific code has already been processed, stop.
    if st.session_state.get(lock_key):
        st.query_params.clear()  # Clean up the URL
        return

    # 5. Immediately set the lock BEFORE any network calls
    st.session_state[lock_key] = True
    
    # 6. Now proceed with the token exchange
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
        
        # FIX: Add state parameter if available
        state = query_params.get("state")
        token_response = flow.fetch_token(
            code=incoming_code,
            state=state
        )
        
        creds = flow.credentials
        
        # Verify the ID token
        idinfo = id_token.verify_oauth2_token(
            creds.id_token,
            requests.Request(),
            os.environ["GOOGLE_CLIENT_ID"],
        )

        # Only update session state on success
        st.session_state["google_email"] = idinfo["email"]
        st.session_state["google_creds"] = creds
        st.session_state["google_logged_in"] = True
        
        # Clear the query parameters
        st.query_params.clear()
        
        # Force a rerun to update the UI
        st.rerun()

    except Exception as e:
        # On failure, clear the lock so the user can try again
        st.session_state.pop(lock_key, None)
        st.error(f"Authentication failed: {str(e)}")
        
        # Log detailed error for debugging
        print(f"Google OAuth Error: {str(e)}")
        print(f"Code used: {incoming_code[:20]}...")
        
        # Clear query params to prevent infinite error loop
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
