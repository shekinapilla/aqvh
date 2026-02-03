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
# Streamlit page config
# -------------------------
st.set_page_config(
    page_title="Quantum Circuit -> Bloch Visualizer",
    page_icon="logo.ico",
    layout="wide"
)

# -------------------------
# Persistent History Storage
# -------------------------
HISTORY_FILE = "history.pkl"

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
        if st.session_state.get("google_logged_in"):
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

# -------------------------
# Helper functions
# -------------------------
def to_matrix_2x2(rho):
    """Converts a density matrix object to a 2x2 numpy array."""
    if hasattr(rho, "data"):
        mat = np.asarray(rho.data)
    elif hasattr(rho, "to_matrix"):
        mat = np.asarray(rho.to_matrix())
    else:
        mat = np.asarray(rho)
    return mat.reshape((2, 2))

def bloch_vector_from_rho_mat(rho_mat):
    """Calculates the Bloch vector from a 2x2 density matrix."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    bx = np.real(np.trace(rho_mat @ X))
    by = np.real(np.trace(rho_mat @ Y))
    bz = np.real(np.trace(rho_mat @ Z))
    return np.array([bx, by, bz])

def purity_from_rho_mat(rho_mat):
    """Calculates the purity of a state from its density matrix."""
    return float(np.real_if_close(np.trace(rho_mat @ rho_mat)))

def plot_bloch_vector(bvec, title="Bloch Sphere"):
    """Generates a 3D plot of a Bloch vector on the Bloch sphere."""
    fig = plt.figure(figsize=(2.5, 2.5), dpi=180)
    ax = fig.add_subplot(111, projection='3d')
    u, v = np.mgrid[0:2*np.pi:80j, 0:np.pi:40j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    ax.plot_surface(xs, ys, zs, alpha=0.12, linewidth=0, color='cyan')
    ax.quiver(0, 0, 0, bvec[0], bvec[1], bvec[2], length=1.0, linewidth=2, color='r')
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    norm = np.linalg.norm(bvec)
    ax.set_title(f"{title}\nvec={np.round(bvec,3)}, |norm|={np.round(norm,3)}", fontsize=8)
    fig.tight_layout()
    return fig

def add_history_entry(name, qc_obj, ops_list):
    """Adds a new circuit to the history session state and saves to disk."""
    st.session_state.history.append(name)
    try:
        st.session_state.saved_circuits.append({
            "name": name,
            "circuit": pickle.loads(pickle.dumps(qc_obj)),
            "ops": ops_list
        })
    except Exception as e:
        st.sidebar.error(f"❌ Failed to save circuit data: {e}")
    save_history_to_disk()

def rebuild_manual_qc():
    """Rebuilds the manual QuantumCircuit object from the list of operations."""
    n = st.session_state.n_qubits
    qc = QuantumCircuit(n)
    skipped = []
    for i, op in enumerate(st.session_state.manual_ops):
        gate = op.get("gate")
        tgt = op.get("target")
        ctrl = op.get("control")
        theta = op.get("theta")
        max_idx = -1
        if tgt is not None:
            max_idx = max(max_idx, tgt)
        if ctrl is not None:
            max_idx = max(max_idx, ctrl)
        if max_idx >= n:
            skipped.append(i)
            continue
        try:
            if gate == "H": qc.h(int(tgt))
            elif gate == "X": qc.x(int(tgt))
            elif gate == "Y": qc.y(int(tgt))
            elif gate == "Z": qc.z(int(tgt))
            elif gate == "S": qc.s(int(tgt))
            elif gate == "T": qc.t(int(tgt))
            elif gate == "CX": qc.cx(int(ctrl), int(tgt))
            elif gate == "CY": qc.cy(int(ctrl), int(tgt))
            elif gate == "CZ": qc.cz(int(ctrl), int(tgt))
            elif gate == "SWAP": qc.swap(int(tgt), int(ctrl))
            elif gate == "RX": qc.rx(float(theta), int(tgt))
            elif gate == "RY": qc.ry(float(theta), int(tgt))
            elif gate == "RZ": qc.rz(float(theta), int(tgt))
            else: skipped.append(i)
        except Exception:
            skipped.append(i)
    st.session_state.manual_qc = qc
    return skipped

def safe_get_qasm(qc: QuantumCircuit):
    """Safely generates QASM 2.0 string from a QuantumCircuit."""
    try:
        return dumps2(qc)
    except Exception as e:
        return f"# Failed to generate QASM: {e}"

def remove_classical_instructions(qc: QuantumCircuit) -> QuantumCircuit:
    """Creates a new circuit containing only the quantum instructions."""
    new_qc = QuantumCircuit(qc.num_qubits)
    for instr, qargs, cargs in qc.data:
        if instr.name not in ["measure", "reset", "barrier"]:
            new_qc.append(instr, qargs, cargs)
    return new_qc

def push_manual_ops_state():
    """Pushes the current manual operations list to its history stack for undo/redo."""
    ops_copy = pickle.loads(pickle.dumps(st.session_state.manual_ops))
    st.session_state.manual_ops_history = st.session_state.manual_ops_history[:st.session_state.manual_ops_pointer+1]
    st.session_state.manual_ops_history.append(ops_copy)
    st.session_state.manual_ops_pointer += 1

def update_circuit_from_qasm_callback():
    """Parses QASM from the text area and updates the uploaded_qc state."""
    edited_qasm = st.session_state.editable_qasm_area
    try:
        updated_qc = QuantumCircuit.from_qasm_str(edited_qasm)
        st.session_state.uploaded_qc = updated_qc
        st.session_state.mode = "qasm"
        st.session_state.editable_qasm = edited_qasm
        if not st.session_state.qasm_history or st.session_state.qasm_history[-1] != edited_qasm:
            st.session_state.qasm_history.append(edited_qasm)
            st.session_state.qasm_redo = []
    except Exception as e:
        st.error(f"❌ Failed to parse edited QASM: {e}")
        st.session_state.uploaded_qc = None

def handle_file_upload_callback():
    """Handles logic when a new file is uploaded."""
    uploaded_file = st.session_state.file_uploader
    if uploaded_file is not None and uploaded_file.name != st.session_state.uploaded_file_name:
        try:
            qasm_string = uploaded_file.getvalue().decode("utf-8")
            st.session_state.uploaded_qc = QuantumCircuit.from_qasm_str(qasm_string)
            st.session_state.mode = "qasm"
            st.session_state.editable_qasm = qasm_string
            st.session_state.qasm_history = [qasm_string]
            st.session_state.qasm_redo = []
            st.session_state.uploaded_file_name = uploaded_file.name
            st.success("✅ QASM file loaded!")
            
        except Exception as e:
            st.error(f"❌ Failed to load QASM: {e}")
            st.session_state.uploaded_qc = None
            st.session_state.mode = "manual"
            st.session_state.uploaded_file_name = None
            

def undo_qasm():
    """Undoes the last QASM change by reverting to the previous state in history."""
    if st.session_state.qasm_history and len(st.session_state.qasm_history) > 1:
        st.session_state.qasm_redo.append(st.session_state.qasm_history.pop())
        st.session_state.editable_qasm = st.session_state.qasm_history[-1]
        try:
            st.session_state.uploaded_qc = QuantumCircuit.from_qasm_str(st.session_state.editable_qasm)
            st.session_state.mode = "qasm"
        except Exception:
            st.session_state.uploaded_qc = None
    

def redo_qasm():
    """Redoes the last QASM undo by moving a state from redo stack to history."""
    if st.session_state.qasm_redo:
        next_qasm = st.session_state.qasm_redo.pop()
        st.session_state.qasm_history.append(next_qasm)
        st.session_state.editable_qasm = next_qasm
        try:
            st.session_state.uploaded_qc = QuantumCircuit.from_qasm_str(st.session_state.editable_qasm)
            st.session_state.mode = "qasm"
        except Exception:
            st.session_state.uploaded_qc = None
    

# -------------------------
# Sidebar
# -------------------------
st.sidebar.image("logo.png", use_column_width=True)
st.sidebar.title("Quantum Visualizer")
st.sidebar.markdown("## Account")

if "google_logged_in" not in st.session_state:
    login_button()
    handle_callback()
else:
    st.sidebar.success("Logged in with Google")



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
