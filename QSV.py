import streamlit as st
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace
from qiskit.qasm2 import dumps as dumps2
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import pickle
import psutil
import gc
import warnings
from google_auth import login_button, handle_callback
from google_auth import upload_history_to_drive
warnings.filterwarnings("ignore")

# -------------------------
# Global Auth State
# -------------------------
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = None   # None | "google" | "guest"

if "google_logged_in" not in st.session_state:
    st.session_state.google_logged_in = False

if "google_email" not in st.session_state:
    st.session_state.google_email = None

# -------------------------
# Streamlit page config - IMPORTANT: Keep this at the top
# -------------------------
st.set_page_config(
    page_title="Quantum Circuit -> Bloch Visualizer",
    page_icon="logo.ico",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide sidebar on login page
)

# -------------------------
# CUSTOM CSS FOR LOGIN PAGE
# -------------------------
login_css = """
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 0;
        padding-bottom: 0;
    }
    
    /* Background animation */
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Login card styling */
    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #0f3443);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        position: relative;
        overflow: hidden;
    }
    
    .login-container::before {
        content: '';
        position: absolute;
        width: 300px;
        height: 300px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(0,247,255,0.1) 0%, transparent 70%);
        top: 10%;
        right: 10%;
        animation: float 6s ease-in-out infinite;
    }
    
    .login-container::after {
        content: '';
        position: absolute;
        width: 200px;
        height: 200px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(148,0,211,0.1) 0%, transparent 70%);
        bottom: 10%;
        left: 10%;
        animation: float 8s ease-in-out infinite reverse;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    .login-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 40px;
        width: 100%;
        max-width: 480px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        position: relative;
        z-index: 1;
        margin: 20px;
        transition: transform 0.3s ease;
    }
    
    .login-card:hover {
        transform: translateY(-5px);
    }
    
    .logo-section {
        text-align: center;
        margin-bottom: 30px;
    }
    
    .logo-circle {
        width: 80px;
        height: 80px;
        background: linear-gradient(135deg, #00f7ff, #9400d3);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 20px;
        font-size: 32px;
        color: white;
        box-shadow: 0 8px 20px rgba(0, 247, 255, 0.3);
    }
    
    .app-title {
        color: white;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 8px;
        background: linear-gradient(to right, #00f7ff, #9400d3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .app-subtitle {
        color: rgba(255, 255, 255, 0.8);
        font-size: 16px;
        margin-bottom: 30px;
        line-height: 1.5;
    }
    
    .divider {
        display: flex;
        align-items: center;
        margin: 30px 0;
    }
    
    .divider::before,
    .divider::after {
        content: '';
        flex: 1;
        height: 1px;
        background: rgba(255, 255, 255, 0.2);
    }
    
    .divider-text {
        color: rgba(255, 255, 255, 0.6);
        padding: 0 15px;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Custom button styling */
    .stButton > button {
        width: 100%;
        border-radius: 12px !important;
        height: 52px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        border: none !important;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 247, 255, 0.3) !important;
    }
    
    .google-btn {
        background: linear-gradient(135deg, #4285f4, #34a853) !important;
        color: white !important;
    }
    
    .guest-btn {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
    }
    
    .btn-content {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 15px;
        margin: 30px 0;
    }
    
    .feature-item {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .feature-item:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateY(-3px);
    }
    
    .feature-icon {
        font-size: 24px;
        margin-bottom: 10px;
        display: block;
    }
    
    .feature-text {
        color: rgba(255, 255, 255, 0.9);
        font-size: 14px;
        font-weight: 500;
    }
    
    .guest-note {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #00f7ff;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
    }
    
    .guest-note p {
        color: rgba(255, 255, 255, 0.8);
        font-size: 14px;
        margin: 0;
        line-height: 1.5;
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
</style>
"""

# -------------------------
# ENHANCED LOGIN PAGE
# -------------------------

# Handle Google callback ONLY if redirected
if "code" in st.query_params:
    handle_callback()
    if st.session_state.get("google_logged_in"):
        st.session_state.auth_mode = "google"

# If user not authenticated yet → show enhanced login page
if st.session_state.auth_mode is None:
    # Apply custom CSS
    st.markdown(login_css, unsafe_allow_html=True)
    
    # Main container
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    
    # Logo and title section
    st.markdown('<div class="logo-section">', unsafe_allow_html=True)
    st.markdown('<div class="logo-circle">🔬</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="app-title">Quantum Visualizer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="app-subtitle">Visualize quantum circuits on the Bloch sphere with real-time simulation and cloud sync</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Features grid
    st.markdown('<div class="feature-grid">', unsafe_allow_html=True)
    st.markdown('''
        <div class="feature-item">
            <span class="feature-icon">🎯</span>
            <div class="feature-text">Build Circuits</div>
        </div>
        <div class="feature-item">
            <span class="feature-icon">🌐</span>
            <div class="feature-text">Bloch Spheres</div>
        </div>
        <div class="feature-item">
            <span class="feature-icon">💾</span>
            <div class="feature-text">Save to Drive</div>
        </div>
        <div class="feature-item">
            <span class="feature-icon">📊</span>
            <div class="feature-text">Export Data</div>
        </div>
    ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Google Login Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Custom button for Google login
        st.markdown('<div class="stButton google-btn">', unsafe_allow_html=True)
        if st.button("🚀 Sign in with Google", key="google_login_main"):
            # We'll use the existing login_button functionality
            pass
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Actually render the Google login button (invisible, triggered by our custom button)
        login_button()
    
    # Divider
    st.markdown('<div class="divider"><span class="divider-text">or</span></div>', unsafe_allow_html=True)
    
    # Guest Mode Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("👤 Continue as Guest", key="guest_login", help="Local storage only, no cloud sync"):
            st.session_state.auth_mode = "guest"
            st.rerun()
    
    # Guest mode note
    st.markdown('<div class="guest-note">', unsafe_allow_html=True)
    st.markdown('<p><strong>ℹ️ Guest Mode:</strong> Your work will be saved locally. For cloud sync and access across devices, sign in with Google.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div style="text-align: center; margin-top: 30px;">', unsafe_allow_html=True)
    st.markdown('<p style="color: rgba(255, 255, 255, 0.5); font-size: 12px;">Quantum computing visualization tool powered by Qiskit</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close login-card
    st.markdown('</div>', unsafe_allow_html=True)  # Close login-container
    
    st.stop()   # ⛔ VERY IMPORTANT: stop app here

# -------------------------
# REST OF YOUR APP (ONLY SHOWN AFTER LOGIN)
# -------------------------

# Show the sidebar now
st.markdown("""
<style>
    /* Re-enable sidebar styling for main app */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2027, #203a43);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# History Storage (Per Mode)
# -------------------------
if st.session_state.auth_mode == "google":
    safe_email = st.session_state.google_email.replace("@", "_").replace(".", "_")
    USER_DIR = os.path.join("user_data", safe_email)
else:
    USER_DIR = os.path.join("user_data", "guest")

os.makedirs(USER_DIR, exist_ok=True)
HISTORY_FILE = os.path.join(USER_DIR, "history.pkl")

def save_history_to_disk():
    try:
        with open(HISTORY_FILE, "wb") as f:
            pickle.dump(
                {
                    "history": st.session_state.history,
                    "saved_circuits": st.session_state.saved_circuits
                },
                f
            )

        # 🔥 Step 3 trigger: upload per user
        if st.session_state.auth_mode == "google":
            upload_history_to_drive(HISTORY_FILE)

    except Exception as e:
        st.sidebar.error(f"Failed saving history to disk: {e}")

def load_history_from_disk():
    """Loads history from the pickle file if it exists."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "rb") as f:
                data = pickle.load(f)
            st.session_state.history = data.get("history", [])
            st.session_state.saved_circuits = data.get("saved_circuits", [])
        except Exception:
            st.session_state.history = []
            st.session_state.saved_circuits = []

# -------------------------
# Initialize session states
# -------------------------
if "initialized" not in st.session_state:
    st.session_state.history = []
    st.session_state.saved_circuits = []
    st.session_state.n_qubits = 1
    st.session_state.manual_ops = []
    st.session_state.manual_qc = QuantumCircuit(1)
    st.session_state.uploaded_qc = None
    st.session_state.mode = "manual"
    st.session_state.editable_qasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];"
    st.session_state.qasm_history = [st.session_state.editable_qasm]
    st.session_state.qasm_redo = []
    st.session_state.manual_ops_history = [[]]
    st.session_state.manual_ops_pointer = 0
    st.session_state.uploaded_file_name = None

    load_history_from_disk()
    st.session_state.initialized = True

# Continue with the rest of your existing code...
# [ALL YOUR EXISTING HELPER FUNCTIONS AND APP CODE GOES HERE]
# ... (paste all your existing helper functions and main app code here)

# -------------------------
# Sidebar (with enhanced styling)
# -------------------------
st.sidebar.image("logo.png", use_column_width=True)
st.sidebar.title("Quantum Visualizer")

# Custom sidebar styling
st.sidebar.markdown("""
<style>
    .sidebar-user-info {
        background: linear-gradient(135deg, rgba(0, 247, 255, 0.1), rgba(148, 0, 211, 0.1));
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 4px solid #00f7ff;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## Account")
st.sidebar.markdown('<div class="sidebar-user-info">', unsafe_allow_html=True)

if st.session_state.auth_mode == "google":
    st.sidebar.success(f"✅ Google: {st.session_state.google_email}")
elif st.session_state.auth_mode == "guest":
    st.sidebar.info("👤 Guest mode (local only)")

st.sidebar.markdown('</div>', unsafe_allow_html=True)

if st.sidebar.button("🚪 Logout", help="Sign out and return to login page"):
    for k in ["auth_mode", "google_logged_in", "google_email", "google_creds"]:
        st.session_state.pop(k, None)
    st.rerun()

def reset_app():
    st.session_state.n_qubits = 1
    st.session_state.manual_ops = []
    st.session_state.manual_qc = QuantumCircuit(1)
    st.session_state.uploaded_qc = None
    st.session_state.mode = "manual"
    st.session_state.editable_qasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];"
    st.session_state.qasm_history = [st.session_state.editable_qasm]
    st.session_state.qasm_redo = []
    st.session_state.manual_ops_history = [[]]
    st.session_state.manual_ops_pointer = 0
    st.session_state.uploaded_file_name = None
    

if st.sidebar.button("🆕 Create New Circuit"):
    reset_app()

# -------------------------
# Load History
# -------------------------
st.sidebar.markdown("### History (Click to load)")
if st.session_state.history:
    for idx, desc in enumerate(st.session_state.history):
        if st.sidebar.button(f"🔹 {desc}", key=f"history_{idx}"):
            try:
                stored = st.session_state.saved_circuits[idx]
                loaded_qc = stored["circuit"]
                loaded_ops = stored.get("ops", [])
                
                # Update all relevant session state variables to match the loaded circuit
                st.session_state.manual_qc = loaded_qc
                st.session_state.manual_ops = loaded_ops
                st.session_state.n_qubits = loaded_qc.num_qubits
                st.session_state.mode = "manual"
                st.session_state.uploaded_qc = None
                st.session_state.uploaded_file_name = None
                
                # Explicitly set the history and pointer for the manual builder to the loaded state
                st.session_state.manual_ops_history = [loaded_ops]
                st.session_state.manual_ops_pointer = 0
                
                # Explicitly update the QASM text area content to reflect the loaded circuit
                new_qasm = safe_get_qasm(loaded_qc)
                st.session_state.editable_qasm = new_qasm
                st.session_state.qasm_history = [new_qasm]
                st.session_state.qasm_redo = []
                
                st.success(f"✅ Loaded: {desc}")
                
            except Exception as e:
                st.sidebar.error(f"❌ Failed to load circuit: {e}")

# -------------------------
# Main Title
# -------------------------
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    padding:18px;
    border-radius:14px;
    text-align:center;
    margin-top:6px;
    margin-bottom:15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.4);
">
    <h1 style="color:#00f7ff; margin:0; font-family:Segoe UI, sans-serif; letter-spacing:1px;">
        Quantum State Visualizer
    </h1>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Top Container: Manual & Upload
# -------------------------
top_container = st.container()
with top_container:
    left_top, right_top = st.columns(2)
    with left_top:
        st.subheader("Manual Gate Builder")
        qubit_display = st.empty()
        qubit_display.markdown(f"### 🔹 Current number of qubits: {st.session_state.n_qubits}")

        manual_input = st.number_input("Enter number of qubits:", min_value=1, max_value=100,
                                         value=st.session_state.n_qubits, step=1, key="manual_n_qubits_input")
        if manual_input != st.session_state.n_qubits:
            st.session_state.n_qubits = int(manual_input)
            st.session_state.manual_ops = []
            st.session_state.manual_ops_history = [[]]
            st.session_state.manual_ops_pointer = 0
            rebuild_manual_qc()
            

        n_qubits = st.session_state.n_qubits

        gate = st.selectbox("Choose Gate", ["H","X","Y","Z","S","T","SWAP","CX","CY","CZ","RX","RY","RZ"])
        target = st.number_input("Target qubit index", 0, max(0, n_qubits-1), 0, key="target_index_input")
        ctrl = None
        if gate in ["CX", "CY", "CZ", "SWAP"]:
            ctrl = st.number_input("Second qubit index", 0, max(0, n_qubits-1), 1, key="control_index_input")
        theta = None
        if gate in ["RX", "RY", "RZ"]:
            theta = st.slider("Rotation θ (radians)", 0.0, 2*np.pi, np.pi/2, key="theta_slider")

        col_add_gate, col_reset_gates = st.columns(2)
        with col_add_gate:
            if st.button("➕ Add Gate to Circuit"):
                op = {"gate": gate, "target": int(target), "control": int(ctrl) if ctrl is not None else None,
                      "theta": float(theta) if theta is not None else None}
                st.session_state.manual_ops.append(op)
                push_manual_ops_state()
                rebuild_manual_qc()
                st.success("✅ Gate added.")
                st.session_state.mode = "manual"
                
        with col_reset_gates:
            if st.button("Clear all Gates"):
                st.session_state.manual_ops = []
                st.session_state.manual_ops_history = [[]]
                st.session_state.manual_ops_pointer = 0
                rebuild_manual_qc()
                st.success("✅ Gates cleared.")
                

        col_manual_undo, col_manual_redo = st.columns(2)
        with col_manual_undo:
            if st.button("↩️ Undo Gate", disabled=st.session_state.manual_ops_pointer <= 0):
                st.session_state.manual_ops_pointer -= 1
                st.session_state.manual_ops = pickle.loads(
                    pickle.dumps(st.session_state.manual_ops_history[st.session_state.manual_ops_pointer])
                )
                rebuild_manual_qc()
                
        with col_manual_redo:
            if st.button("↪️ Redo Gate", disabled=st.session_state.manual_ops_pointer >= len(st.session_state.manual_ops_history)-1):
                st.session_state.manual_ops_pointer += 1
                st.session_state.manual_ops = pickle.loads(
                    pickle.dumps(st.session_state.manual_ops_history[st.session_state.manual_ops_pointer])
                )
                rebuild_manual_qc()
                

        st.markdown("---")
        circuit_name = st.text_input("Name your circuit", placeholder="e.g. Bell State", key="circuit_name_input")
        if st.button("💾 Save Circuit to History"):
            name = circuit_name.strip() if circuit_name.strip() else f"Manual Circuit ({n_qubits} qubits)"
            add_history_entry(name, st.session_state.manual_qc, st.session_state.manual_ops.copy())
            st.success(f"✅ Saved: {name}")

        st.markdown("### 🧱 Current Gate Operations")
        for idx, op in enumerate(st.session_state.manual_ops):
            gate_str = f"**{idx+1}. {op['gate']}** on qubit `{op['target']}`"
            if op.get("control") is not None:
                gate_str += f", ctrl: `{op['control']}`"
            if op.get("theta") is not None:
                gate_str += f", θ=`{op['theta']:.2f}`"

            cols = st.columns([0.85, 0.15])
            cols[0].markdown(gate_str)
            if cols[1].button("❌", key=f"delete_gate_{idx}", help="Remove this gate"):
                push_manual_ops_state()
                st.session_state.manual_ops.pop(idx)
                rebuild_manual_qc()
                st.rerun()

    with right_top:
        st.subheader("Upload QASM / Edit & Download")
        
        uploaded_file = st.file_uploader("Upload QASM 2.0 file", type=["qasm", "txt"], key="file_uploader", on_change=handle_file_upload_callback)
        
        current_qc = st.session_state.uploaded_qc if st.session_state.mode == "qasm" else st.session_state.manual_qc

        if isinstance(current_qc, QuantumCircuit) and current_qc.num_qubits > 0:
            if st.session_state.mode == "manual":
                new_qasm = safe_get_qasm(current_qc)
                if new_qasm != st.session_state.editable_qasm:
                    st.session_state.editable_qasm = new_qasm
                    st.session_state.qasm_history.append(new_qasm)
                    st.session_state.qasm_redo = []
            
            st.text_area(
                "✏️ Edit QASM here to update the circuit:",
                value=st.session_state.editable_qasm,
                height=250,
                key="editable_qasm_area",
                on_change=update_circuit_from_qasm_callback
            )

            col_q1, col_q2 = st.columns([0.3, 0.7])
            with col_q1:
                st.button("↩️ Undo QASM", on_click=undo_qasm, disabled=len(st.session_state.qasm_history) <= 1)
            with col_q2:
                st.button("↪️ Redo QASM", on_click=redo_qasm, disabled=not st.session_state.qasm_redo)

            st.download_button(
                "⬇ Download QASM",
                data=st.session_state.editable_qasm,
                file_name="quantum_circuit.qasm",
                mime="text/plain"
            )
        else:
            st.info("Build a circuit manually or upload a QASM file to see the code here.")

# -------------------------
# Output / Visualization
# -------------------------
st.markdown("---")
bottom_container = st.container()
with bottom_container:

    qc_to_display = st.session_state.uploaded_qc if st.session_state.mode == "qasm" else st.session_state.manual_qc

    if qc_to_display and qc_to_display.num_qubits > 0:

        st.header("Circuit Diagram & State Visualization")

        # ===========================
        # CIRCUIT DIAGRAM
        # ===========================
        st.write("### Quantum Circuit Diagram")
        try:
            fig_circ = qc_to_display.draw(output="mpl", style="iqp", scale=0.8)
            buf_circ = io.BytesIO()
            fig_circ.savefig(buf_circ, format="png", dpi=200, bbox_inches="tight")
            plt.close(fig_circ)
            st.image(buf_circ.getvalue())

            st.download_button(
                "⬇ Download Circuit Image",
                data=buf_circ.getvalue(),
                file_name="quantum_circuit.png",
                mime="image/png"
            )

            del buf_circ

        except Exception as e:
            st.warning(f"Could not draw circuit diagram: {e}")

        # ===========================
        # BLOCH SPHERE VISUALIZATION
        # ===========================
        st.write("### Per-Qubit Bloch Vectors & Purity")


        # Remove classical ops
        quantum_only_qc = remove_classical_instructions(qc_to_display)

        # --- Compute statevector safely ---
        try:
            if not quantum_only_qc.data:
                state = Statevector.from_int(0, 2**quantum_only_qc.num_qubits)
            else:
                state = Statevector.from_instruction(quantum_only_qc)
        except Exception as e:
            st.warning(f"⚠️ Unable to simulate circuit: {e}")
            st.stop()

        # Convert to numpy
        psi = np.asarray(state.data, dtype=complex)
        n = quantum_only_qc.num_qubits

        # Try reshaping
        try:
            psi_tensor = psi.reshape([2]*n)
        except Exception:
            st.warning("⚠️ Insufficient memory to reshape statevector.")
            st.stop()

        # Layout columns
        num_cols = min(n, 2)
        cols = st.columns(num_cols)

        def has_memory_for_qubit(n):
            """Check if memory is enough to compute reduced density."""
            needed_bytes = (2**(n-1)) * 2 * 16  # complex128
            free_bytes = psutil.virtual_memory().available
            safe = needed_bytes < free_bytes * 0.4
            return safe, needed_bytes, free_bytes

        # ===========================
        # MAIN BLOCH LOOP
        # ===========================
        for q in range(n):

            safe, need, free = has_memory_for_qubit(n)
            need_mb = need / (1024**2)
            free_mb = free / (1024**2)

            if not safe:
                st.warning(
                    f"⚠️ Stopped at qubit {q} due to insufficient memory.\n"
                    f"Computed Bloch vectors up to qubit {q-1}.\n"
                    f"(Needs ~{need_mb:.1f} MB but only {free_mb:.1f} MB free.)"
                )
                break

            with cols[q % num_cols]:

                try:
                    # Efficient reduced density matrix for pure states
                    psi_perm = np.moveaxis(psi_tensor, q, 0).reshape(2, -1)
                    rho_mat = psi_perm @ psi_perm.conj().T

                    # Bloch & Purity
                    bvec = bloch_vector_from_rho_mat(rho_mat)
                    purity = purity_from_rho_mat(rho_mat)
                    norm = np.linalg.norm(bvec)

                    st.write(f"**Qubit {q}**")
                    st.code(
                        f"Bloch = {np.round(bvec,4)}\n"
                        f"Norm = {norm:.4f}, Purity = {purity:.4f}"
                    )

                    # Bloch Sphere Plot
                    fig = plot_bloch_vector(bvec, title=f"Qubit {q}")
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    st.image(buf.getvalue())

                    # Download
                    st.download_button(
                        f"⬇ Q{q} Bloch",
                        data=buf.getvalue(),
                        file_name=f"bloch_qubit{q}.png",
                        mime="image/png",
                        key=f"download_bloch_{q}"
                    )

                    del buf, fig, psi_perm

                except MemoryError:
                    st.warning(
                        f"⚠️ Stopped at qubit {q}: system ran out of memory.\n"
                        f"Computed Bloch vectors up to qubit {q-1}."
                    )
                    break

                except Exception as e:
                    st.warning(
                        f"⚠️ Stopped at qubit {q}: {e}\n"
                        f"Computed Bloch vectors up to qubit {q-1}."
                    )
                    break

        gc.collect()

    else:
        st.info("Create or upload a circuit to display outputs.")
