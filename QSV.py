# -------------------------
# Imports
# -------------------------
import streamlit as st

# --- Qiskit core ---
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, partial_trace
from qiskit.qasm2 import dumps as dumps2


# --- IBM Runtime (CORRECT for 2025/2026) ---
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit import transpile






# --- Numerical & plotting ---
import numpy as np
import matplotlib.pyplot as plt

# --- IO & persistence ---
import io
import os
import pickle
import zipfile
from datetime import datetime

# --- Typing (clarity) ---
from typing import List, Dict, Optional


# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(
    page_title="Quantum Circuit Composer & Visualizer",
    page_icon="⚛️",  # circuit/quantum themed, avoids file-path issues
    layout="wide",
    initial_sidebar_state="expanded"
)



# -------------------------
# Persistent History Storage
# -------------------------

HISTORY_FILE = "history.pkl"
MAX_HISTORY = 20


def save_history_to_disk():
    """Persist saved circuits (QASM-based) to disk."""
    try:
        with open(HISTORY_FILE, "wb") as f:
            pickle.dump(st.session_state.saved_circuits, f)
    except Exception as e:
        st.sidebar.error(f"❌ Failed saving history: {e}")


def load_history_from_disk():
    """Load saved circuits from disk."""
    if not os.path.exists(HISTORY_FILE):
        st.session_state.saved_circuits = []
        return

    try:
        with open(HISTORY_FILE, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, list):
            st.session_state.saved_circuits = data[:MAX_HISTORY]
        else:
            st.session_state.saved_circuits = []

    except Exception:
        st.session_state.saved_circuits = []


def add_circuit_to_history(name: str):
    """
    Save the current circuit into history (QASM-based).
    Acts like calculator memory (max 20 entries).
    """
    qasm = st.session_state.editable_qasm

    entry = {
        "name": name,
        "qasm": qasm,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

    # Insert newest at top
    st.session_state.saved_circuits.insert(0, entry)

    # Enforce max history size
    st.session_state.saved_circuits = st.session_state.saved_circuits[:MAX_HISTORY]

    save_history_to_disk()


def load_circuit_from_history(index: int):
    """
    Load a circuit from history into the main app.
    Fully syncs QASM editor, circuit, qubit count, and output.
    """
    try:
        entry = st.session_state.saved_circuits[index]
        qasm = entry["qasm"]

        qc = QuantumCircuit.from_qasm_str(qasm)

        # 🔥 Single source of truth
        st.session_state.current_qc = qc
        st.session_state.manual_qc = qc
        st.session_state.manual_ops = []  # history is QASM-driven

        # 🔥 Sync QASM editor
        st.session_state.editable_qasm = qasm
        st.session_state.last_qasm = qasm
        st.session_state.qasm_source = "history"

        # 🔥 Sync qubit count
        st.session_state.n_qubits = qc.num_qubits

        st.success(f"✅ Loaded: {entry['name']}")

    except Exception as e:
        st.error(f"❌ Failed to load circuit: {e}")



# -------------------------
# Helper Functions (FINAL – SINGLE SOURCE OF TRUTH)
# -------------------------

# =================================================
# Bloch / State helpers
# =================================================

def to_matrix_2x2(rho):
    if hasattr(rho, "data"):
        mat = np.asarray(rho.data)
    elif hasattr(rho, "to_matrix"):
        mat = np.asarray(rho.to_matrix())
    else:
        mat = np.asarray(rho)
    return mat.reshape((2, 2))


def bloch_vector_from_rho_mat(rho_mat):
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    return np.array([
        np.real(np.trace(rho_mat @ X)),
        np.real(np.trace(rho_mat @ Y)),
        np.real(np.trace(rho_mat @ Z)),
    ])


def purity_from_rho_mat(rho_mat):
    return float(np.real_if_close(np.trace(rho_mat @ rho_mat)))


def plot_bloch_vector(bvec, title="Bloch Sphere"):
    fig = plt.figure(figsize=(2.5, 2.5), dpi=180)
    ax = fig.add_subplot(111, projection="3d")

    u, v = np.mgrid[0:2*np.pi:80j, 0:np.pi:40j]
    ax.plot_surface(
        np.cos(u) * np.sin(v),
        np.sin(u) * np.sin(v),
        np.cos(v),
        alpha=0.12,
        linewidth=0,
        color="cyan"
    )

    ax.quiver(0, 0, 0, bvec[0], bvec[1], bvec[2],
              length=1.0, linewidth=2, color="r")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


# =================================================
# QASM helpers
# =================================================

def safe_get_qasm(qc: QuantumCircuit) -> str:
    try:
        return dumps2(qc)
    except Exception as e:
        return f"# QASM generation failed: {e}"


def remove_classical_instructions(qc: QuantumCircuit) -> QuantumCircuit:
    new_qc = QuantumCircuit(qc.num_qubits)
    for instr, qargs, _ in qc.data:
        if instr.name not in ("measure", "reset", "barrier"):
            new_qc.append(instr, qargs)
    return new_qc


# =================================================
# Manual Builder (ONLY builds circuit)
# =================================================

def manual_qc():
    qc_old = st.session_state.current_qc
    n = qc_old.num_qubits

    qc = QuantumCircuit(n, n)

    for op in st.session_state.manual_ops:
        g = op["gate"]
        t = op.get("target")
        c = op.get("control")
        th = op.get("theta")

        if g == "H": qc.h(t)
        elif g == "X": qc.x(t)
        elif g == "Y": qc.y(t)
        elif g == "Z": qc.z(t)
        elif g == "S": qc.s(t)
        elif g == "T": qc.t(t)
        elif g == "RX": qc.rx(th, t)
        elif g == "RY": qc.ry(th, t)
        elif g == "RZ": qc.rz(th, t)
        elif g == "CX": qc.cx(c, t)
        elif g == "CY": qc.cy(c, t)
        elif g == "CZ": qc.cz(c, t)
        elif g == "SWAP": qc.swap(t, c)
        elif g == "MEASURE": qc.measure(t, t)

    st.session_state.current_qc = qc
    st.session_state.manual_qc = qc
    st.session_state.qasm_source = "manual"
    sync_from_circuit()


# =================================================
# Auto-sync (NO LOOPS)
# =================================================

def sync_from_circuit():
    qc = st.session_state.current_qc
    qasm = safe_get_qasm(qc)

    if qasm != st.session_state.last_qasm:
        st.session_state.editable_qasm = qasm
        st.session_state.last_qasm = qasm


def sync_from_qasm():
    qasm = st.session_state.editable_qasm

    if qasm == st.session_state.last_qasm:
        return

    try:
        qc = QuantumCircuit.from_qasm_str(qasm)

        st.session_state.current_qc = qc
        st.session_state.manual_qc = qc
        st.session_state.manual_ops = []

        st.session_state.qasm_source = "editor"
        st.session_state.last_qasm = qasm

    except Exception:
        pass


# =================================================
# IBM Quantum helpers (connection + execution)
# =================================================

def connect_ibm_quantum(token: str, instance: str):
    try:
        service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=token,
            instance=instance
        )

        backends = service.backends()

        st.session_state.ibm_service = service
        st.session_state.ibm_backends = backends
        st.session_state.ibm_connected = True

        st.sidebar.success("✅ Connected to IBM Quantum")

    except Exception as e:
        st.session_state.ibm_connected = False
        st.sidebar.error(f"❌ IBM connection failed: {e}")

def run_on_ibm_backend(qc: QuantumCircuit, backend_name: str):
    service = st.session_state.ibm_service
    backend = service.backend(backend_name)

    qc_exec = qc.copy()

    # Ensure measurements
    if not qc_exec.clbits:
        qc_exec.measure_all()

    # Transpile for hardware
    tqc = transpile(qc_exec, backend)

    # ✅ Correct Sampler (2025+ API)
    sampler = Sampler(mode=backend)

    job = sampler.run([tqc], shots=1024)
    result = job.result()

    # ✅ FINAL correct way to get counts
    counts = result[0].data["c"].get_counts()

    st.session_state.last_run_result = counts
    st.session_state.last_run_mode = "hardware"


# -------------------------
# Initialize session state (FINAL — HARD LOCK)
# -------------------------
if "initialized" not in st.session_state:

    # =================================================
    # Core circuit state (SINGLE SOURCE OF TRUTH)
    # =================================================
    qc = QuantumCircuit(1, 1)  # 1 qubit + 1 classical bit

    st.session_state.current_qc = qc
    st.session_state.manual_qc = qc
    st.session_state.manual_ops = []

    # =================================================
    # Gate palette UI state
    # =================================================
    st.session_state.selected_gate = None

    # =================================================
    # QASM authority
    # manual | editor | upload | history
    # =================================================
    st.session_state.qasm_source = "manual"

    initial_qasm = safe_get_qasm(qc)
    st.session_state.editable_qasm = initial_qasm
    st.session_state.last_qasm = initial_qasm

    # =================================================
    # Execution / backend
    # =================================================
    st.session_state.run_target = "Local Simulator"

    # =================================================
    # History
    # =================================================
    st.session_state.saved_circuits = []
    load_history_from_disk()

    # =================================================
    # IBM Quantum state
    # =================================================
    st.session_state.ibm_connected = False
    st.session_state.ibm_service = None
    st.session_state.ibm_backends = []
    st.session_state.selected_ibm_backend = None
    st.session_state.last_run_result = None
    st.session_state.last_run_mode = None

    # =================================================
    # Final lock
    # =================================================
    st.session_state.initialized = True



    
# -------------------------
# Sidebar (Execution + History)
# -------------------------

st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.title("Quantum Composer")

# =========================================================
# ⚙️ Execution Target (Palette-style)
# =========================================================
st.sidebar.markdown("### ⚙️ Run On")

BACKENDS = [
    "Local Simulator",
    "IBM Simulator",
    "IBM Real Hardware"
]

def select_backend(b):
    st.session_state.run_target = b

# initialize if missing
if "run_target" not in st.session_state:
    st.session_state.run_target = "Local Simulator"

for b in BACKENDS:
    active = st.session_state.run_target == b
    st.sidebar.button(
        b,
        key=f"backend_{b}",
        use_container_width=True,
        type="primary" if active else "secondary",
        on_click=select_backend,
        args=(b,)
    )

if st.session_state.run_target != "Local Simulator":
    st.sidebar.info("🔒 IBM backend integration will be enabled after login setup.")
if st.session_state.run_target in ["IBM Simulator", "IBM Real Hardware"]:

    if st.sidebar.button("▶ Run on IBM Backend", key="run_ibm"):
        if not st.session_state.get("selected_backend"):
            st.sidebar.error("Select an IBM backend first.")
        else:
            run_on_ibm_backend(
                st.session_state.current_qc,
                st.session_state.selected_backend
            )
            st.sidebar.success("Job submitted to IBM backend.")

# =========================================================
# 🆕 Reset Circuit
# =========================================================
def reset_app():
    qc = QuantumCircuit(1, 1)  # 1 qubit + 1 classical bit

    # 🔥 core reset (SINGLE SOURCE OF TRUTH)
    st.session_state.current_qc = qc
    st.session_state.manual_qc = qc
    st.session_state.manual_ops = []

    # 🔥 QASM reset
    qasm = safe_get_qasm(qc)
    st.session_state.editable_qasm = qasm
    st.session_state.last_qasm = qasm
    st.session_state.qasm_source = "manual"

    # 🔥 UI reset
    st.session_state.selected_gate = None
    st.session_state.qasm_upload_applied = False

    st.success("✅ New circuit created.")


# =========================================================
# 📁 History (Calculator-style, max 20)
# =========================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Saved Circuits (max 20)")

if not st.session_state.saved_circuits:
    st.sidebar.caption("No saved circuits yet.")
else:
    for idx, entry in enumerate(st.session_state.saved_circuits):
        label = f"{entry['name']}  ⏱ {entry['timestamp']}"
        if st.sidebar.button(label, key=f"load_hist_{idx}"):
            load_circuit_from_history(idx)
#---------------------------------------------------------
# IBM SIDEBAR
#---------------------------------------------------------
st.sidebar.markdown("### ☁️ IBM Quantum")

# --- Token input ---
ibm_token = st.sidebar.text_input(
    "IBM Quantum API Token",
    type="password",
    key="ibm_api_token"
)

# --- Instance input ---
instance = st.sidebar.text_input(
    "IBM Quantum Instance (hub/group/project)",
    placeholder="ibm-q/open/main",
    key="ibm_instance"
)

# --- Connect button (ONLY ONE) ---
if st.sidebar.button(
    "🔑 Connect to IBM Quantum",
    key="connect_ibm_quantum_btn"
):
    connect_ibm_quantum(ibm_token, instance)


# --- Backend display ---
if st.session_state.get("ibm_connected", False):

    if not st.session_state.get("ibm_backends"):
        st.sidebar.warning("No IBM backends available for this account.")
    else:
        backend_names = [b.name for b in st.session_state.ibm_backends]

        st.sidebar.markdown("### 🖥️ IBM Backends")

        st.session_state.selected_backend = st.sidebar.selectbox(
            "Select IBM backend",
            backend_names,
            key="ibm_backend_select"
        )






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
        Quantum Circuit Composer
    </h1>
    <p style="color:#b8e9ff; margin-top:6px; font-size:14px;">
        Manual Gate Palette · Live QASM · Visualization · IBM Backends
    </p>
</div>
""", unsafe_allow_html=True)


# -------------------------
# Top Container: Manual Builder & QASM
# -------------------------
top_container = st.container()
with top_container:
    left_top, right_top = st.columns(2)

    # =========================================================
    # LEFT: Manual Gate Builder
    # =========================================================
    with left_top:
        st.subheader("🧠 Manual Gate Builder")

        # ---------- Qubit Count (single source of truth) ----------
        actual_qubits = st.session_state.current_qc.num_qubits
        st.markdown(f"**Current qubits:** {actual_qubits}")

        n = st.number_input(
            "Number of qubits (manual mode only)",
            min_value=1,
            max_value=100,
            value=actual_qubits,
            step=1,
            key="qubit_input"
        )

        if n != actual_qubits and st.session_state.qasm_source == "manual":
            qc = QuantumCircuit(n, n)
            st.session_state.current_qc = qc
            st.session_state.manual_qc = qc
            st.session_state.manual_ops = []
            sync_from_circuit()
            st.rerun()

        # ---------------------------------------------------------
        # 🎨 Gate Palette
        # ---------------------------------------------------------
        st.markdown("### Gate Palette")

        SINGLE = ["H", "X", "Y", "Z", "S", "T"]
        ROT = ["RX", "RY", "RZ"]
        TWO = ["CX", "CY", "CZ", "SWAP"]

        def select_gate(g):
            st.session_state.selected_gate = g

        def gate_btn(g):
            st.button(
                g,
                key=f"gate_{g}",
                use_container_width=True,
                on_click=select_gate,
                args=(g,),
                type="primary" if st.session_state.selected_gate == g else "secondary"
            )

        st.markdown("**Single Qubit**")
        for c, g in zip(st.columns(6), SINGLE):
            with c:
                gate_btn(g)

        st.markdown("**Rotation**")
        for c, g in zip(st.columns(3), ROT):
            with c:
                gate_btn(g)

        st.markdown("**Two Qubit**")
        for c, g in zip(st.columns(4), TWO):
            with c:
                gate_btn(g)

        st.markdown("**Measurement**")
        gate_btn("MEASURE")

        # ---------------------------------------------------------
        # 🎛️ Gate Parameters
        # ---------------------------------------------------------
        if st.session_state.selected_gate:
            st.markdown("---")
            st.markdown(f"### Parameters for `{st.session_state.selected_gate}`")

            max_q = st.session_state.current_qc.num_qubits - 1

            tgt = st.number_input(
                "Target qubit",
                min_value=0,
                max_value=max_q,
                value=0,
                key="tgt_qubit"
            )

            ctrl = None
            theta = None

            if st.session_state.selected_gate in TWO:
                ctrl = st.number_input(
                    "Control qubit",
                    min_value=0,
                    max_value=max_q,
                    value=0,
                    key="ctrl_qubit"
                )

            if st.session_state.selected_gate in ROT:
                theta = st.slider(
                    "Rotation θ (radians)",
                    0.0,
                    2 * np.pi,
                    np.pi / 2,
                    key="theta_slider"
                )

            if st.button("➕ Apply Gate", use_container_width=True):
                st.session_state.manual_ops.append({
                    "gate": st.session_state.selected_gate,
                    "target": int(tgt),
                    "control": int(ctrl) if ctrl is not None else None,
                    "theta": float(theta) if theta is not None else None
                })

                manual_qc()
                st.session_state.qasm_source = "manual"
                st.session_state.qasm_upload_applied = False
                st.session_state.selected_gate = None
                st.rerun()

        # ---------------------------------------------------------
        # 💾 Save Circuit (ALWAYS VISIBLE)
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("💾 Save Circuit")

        circuit_name = st.text_input(
            "Circuit name",
            placeholder="e.g. Bell State",
            key="save_circuit_name"
        )

        if st.button("💾 Save to History", use_container_width=True):
            q = st.session_state.current_qc.num_qubits
            name = circuit_name.strip() or f"Circuit ({q} qubits)"
            add_circuit_to_history(name)
            st.success(f"✅ Saved: {name}")

    # =========================================================
    # RIGHT: QASM Builder (ALWAYS VISIBLE)
    # =========================================================
    with right_top:
        st.subheader("🧾 QASM Builder")

        # -----------------------------------------------------
        # Upload QASM (apply ONCE)
        # -----------------------------------------------------
        uploaded = st.file_uploader(
            "Upload QASM file",
            type=["qasm", "txt"],
            key="qasm_upload"
        )

        if uploaded and not st.session_state.qasm_upload_applied:
            try:
                qasm_text = uploaded.getvalue().decode("utf-8")
                qc = QuantumCircuit.from_qasm_str(qasm_text)

                st.session_state.current_qc = qc
                st.session_state.manual_qc = qc
                st.session_state.manual_ops = []

                st.session_state.editable_qasm = qasm_text
                st.session_state.last_qasm = qasm_text
                st.session_state.qasm_source = "upload"
                st.session_state.qasm_upload_applied = True

                st.success("✅ QASM loaded and applied")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Invalid QASM file: {e}")

        # -----------------------------------------------------
        # QASM Editor (LIVE)
        # -----------------------------------------------------
        st.text_area(
            "Edit QASM (live sync)",
            value=st.session_state.editable_qasm,
            height=320,
            key="editable_qasm",
            on_change=sync_from_qasm
        )

        # -----------------------------------------------------
        # Manual → QASM sync
        # -----------------------------------------------------
        if st.session_state.qasm_source == "manual":
            generated = safe_get_qasm(st.session_state.current_qc)
            if generated != st.session_state.last_qasm:
                st.session_state.editable_qasm = generated
                st.session_state.last_qasm = generated

        # -----------------------------------------------------
        # Download
        # -----------------------------------------------------
        st.download_button(
            "⬇ Download QASM",
            data=st.session_state.editable_qasm,
            file_name="quantum_circuit.qasm",
            mime="text/plain",
            use_container_width=True
        )



# -------------------------
# Output / Visualization
# -------------------------
st.markdown("---")

qc_to_display = st.session_state.current_qc
if "last_run_mode" not in st.session_state:
    st.session_state.last_run_mode = None

# =========================================================
# 🧪 HARDWARE MODE OUTPUT
# =========================================================
if st.session_state.last_run_mode == "hardware":

    st.header("📊 IBM Hardware Results")

    counts = st.session_state.last_run_result

    clean_counts = {
    k: int(v * 1024) if isinstance(v, float) else int(v)
    for k, v in counts.items()
}


    st.json(clean_counts)
    st.bar_chart(clean_counts)

    st.info("⚠️ Bloch vectors are not available on real quantum hardware.")

# =========================================================
# 🧠 SIMULATOR / MANUAL MODE OUTPUT
# =========================================================
elif qc_to_display and qc_to_display.num_qubits > 0:

    st.header("📈 Circuit Output & Visualization")

    zip_files = {}

    # -------------------------
    # Circuit Diagram
    # -------------------------
    st.subheader("🔗 Quantum Circuit Diagram")

    try:
        fig = qc_to_display.draw(output="mpl", scale=0.8)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        circuit_png = buf.getvalue()
        zip_files["circuit.png"] = circuit_png

        st.image(circuit_png)

        st.download_button(
            "⬇ Download Circuit Diagram",
            data=circuit_png,
            file_name="quantum_circuit.png",
            mime="image/png"
        )

    except Exception as e:
        st.warning(f"Could not draw circuit diagram: {e}")

    # -------------------------
    # Bloch Vectors
    # -------------------------
    st.subheader("🧭 Bloch Vectors (Per Qubit)")

    try:
        quantum_only_qc = remove_classical_instructions(qc_to_display)

        if quantum_only_qc.data:
            state = Statevector.from_instruction(quantum_only_qc)
        else:
            state = Statevector.from_int(0, 2**quantum_only_qc.num_qubits)

        n = quantum_only_qc.num_qubits
        cols = st.columns(min(2, n))

        for q in range(n):
            with cols[q % len(cols)]:
                reduced = partial_trace(state, [i for i in range(n) if i != q])
                rho = to_matrix_2x2(reduced)

                bvec = bloch_vector_from_rho_mat(rho)
                purity = purity_from_rho_mat(rho)

                st.markdown(f"**Qubit {q}**")
                st.code(
                    f"Bloch = {np.round(bvec, 4)}\nPurity = {purity:.4f}",
                    language=None
                )

                fig = plot_bloch_vector(bvec, title=f"Qubit {q}")
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                plt.close(fig)

                img = buf.getvalue()
                zip_files[f"bloch_qubit_{q}.png"] = img

                st.image(img)

                st.download_button(
                    f"⬇ Download Bloch Q{q}",
                    data=img,
                    file_name=f"bloch_qubit_{q}.png",
                    mime="image/png",
                    key=f"bloch_dl_{q}"
                )

    except Exception as e:
        st.warning(f"Bloch visualization skipped: {e}")

    # -------------------------
    # ZIP Download
    # -------------------------
    st.markdown("---")
    st.subheader("📦 Download All Outputs")

    zip_files["circuit.qasm"] = st.session_state.editable_qasm.encode()

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in zip_files.items():
            zf.writestr(name, data)

    st.download_button(
        "⬇ Download ALL (ZIP)",
        data=zip_buf.getvalue(),
        file_name="quantum_outputs.zip",
        mime="application/zip",
        use_container_width=True
    )

else:
    st.info("Create or load a circuit to see outputs.")
 



