import streamlit as st
from google_auth import start_google_login


def render_login_page():

    # ---------- Layout ----------
    st.markdown(
        """
        <div style="text-align:center; margin-bottom:20px;">
            <h1>Quantum State Visualizer</h1>
            <p>Please login to continue</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # ---------- Local login ----------
        with st.form("login_form_main"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            login_btn = st.form_submit_button("Login")

            if login_btn:
                if email and password:
                    st.session_state.is_authenticated = True
                    st.session_state.auth_mode = "local"
                    st.rerun()
                else:
                    st.error("Email and password required")

        st.markdown("---")

        # ---------- Google login ----------
        if st.button("Continue with Google", use_container_width=True):
            start_google_login()

        st.markdown("---")

        # ---------- Guest login ----------
        if st.button("Continue without login", use_container_width=True):
            st.session_state.is_authenticated = True
            st.session_state.auth_mode = "guest"
            st.rerun()
