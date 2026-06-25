"""
Experiment III — Hardware validation on IBM ibm_kobe.

Compiles a trained near-identity residual core into the zero-shot subspace
circuit V = U_A S (state preparation S|0> = |x_sub> followed by the subspace
unitary U_A = e^{-iH}), runs it on real hardware, and compares the measured
output distribution to the ideal one (Hellinger fidelity, total-variation
distance).  Sweeps the subspace rank k in {2,4,8,16} (= 1..4 qubits) to expose
how the hardware fidelity tracks the size of the compiled subspace unitary.

  from src.experiments.exp3_hardware import run, plot, load    # run(submit=True) uses device time
"""
import os
import json

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

from src import RESULT_DIR, SEED, SHOTS, BACKEND_NAME
from src.core import hardware
from src.analysis import plotting
from src.analysis.metrics import hellinger_fidelity, tvd
from src.experiments.validation import ClassicalResNet, train_model, DEVICE
from src.core.data import mnist_split_loaders
from src.core.geometric_qml import transfer_map

_EXPDIR = os.path.join(RESULT_DIR, "exp3_hardware")
os.makedirs(_EXPDIR, exist_ok=True)

K_LIST = [2, 4, 8, 16]      # subspace dims -> 1,2,3,4 qubits
N_INPUTS = 2                # inputs per k
_FIG_STEM = "exp3_fidelity"


# --------------------------------------------------------------------------
# Circuit construction
# --------------------------------------------------------------------------
def _clean_unitary(M):
    x, _, vh = np.linalg.svd(M.astype(np.complex128))
    return x @ vh


def _state_prep_unitary(x):
    """Unitary S with S|0> = x (Gram-Schmidt completion of x to an ONB)."""
    nd = len(x)
    cols = [x.astype(np.complex128) / np.linalg.norm(x)]
    eye = np.eye(nd, dtype=np.complex128)
    for i in range(nd):
        v = eye[:, i].copy()
        for c in cols:
            v = v - (np.conj(c) @ v) * c
        nv = np.linalg.norm(v)
        if nv > 1e-9:
            cols.append(v / nv)
        if len(cols) == nd:
            break
    return np.column_stack(cols)


def make_circuit(state_vec, U):
    """Single n-qubit unitary V = U_A S, so V|0> = U_A|x_sub>."""
    n = int(np.log2(len(state_vec)))
    V = _clean_unitary(U.astype(np.complex128) @ _state_prep_unitary(state_vec))
    qc = QuantumCircuit(n)
    qc.append(UnitaryGate(V), range(n))
    return qc


def ideal_distribution(state_vec, U):
    return np.asarray(Statevector(make_circuit(state_vec, U)).probabilities())


def build_jobs():
    """Train once; for each k produce (x_subs, U_A), circuits, and ideal dists."""
    torch.manual_seed(SEED)
    train_all, _, _, test = mnist_split_loaders()
    model = train_model(ClassicalResNet(hidden_dim=64).to(DEVICE), train_all, epochs=2)
    W = model.core_layer.weight.data.clone().cpu()

    imgs, _ = next(iter(test))
    imgs = imgs[:N_INPUTS].to(DEVICE)
    with torch.no_grad():
        h = model.relu(model.fc_in(model.flatten(imgs))).to(torch.complex64).cpu()

    jobs = []
    for k in K_LIST:
        Q, U_A, _ = transfer_map(W, k)
        U_np = _clean_unitary(U_A.numpy())
        x_sub = (h @ Q.conj()).numpy()
        x_sub = x_sub / (np.linalg.norm(x_sub, axis=1, keepdims=True) + 1e-12)
        for j in range(N_INPUTS):
            jobs.append({"k": k, "n": int(np.log2(k)), "input_idx": j,
                         "x_sub": x_sub[j], "U_A": U_np,
                         "ideal": ideal_distribution(x_sub[j], U_np)})
    return jobs


# --------------------------------------------------------------------------
# Run (Aer dry-run or ibm_kobe submit)
# --------------------------------------------------------------------------
def run(submit=False):
    jobs = build_jobs()
    circuits = []
    for jb in jobs:
        qc = make_circuit(jb["x_sub"], jb["U_A"]); qc.measure_all()
        circuits.append(qc)

    if submit:
        backend = hardware.get_backend(BACKEND_NAME)
        backend_name = backend.name
    else:
        from qiskit_aer import AerSimulator
        backend = AerSimulator(); backend_name = "AerSimulator (dry-run)"

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    isa = [pm.run(c) for c in circuits]
    for jb, c in zip(jobs, isa):
        jb["twoq"] = c.num_nonlocal_gates()

    if submit:
        print(f"Submitting {len(isa)} circuits x {SHOTS} shots to {backend_name}; "
              f"2q-gates per circuit: {[jb['twoq'] for jb in jobs]}")
        res = hardware.run_sampler(isa, backend, shots=SHOTS)
        hw_dists = [hardware.counts_to_probs(hardware.get_counts(res, i), 2 ** jobs[i]["n"], SHOTS)
                    for i in range(len(jobs))]
    else:
        res = backend.run(isa, shots=SHOTS).result()
        hw_dists = [hardware.counts_to_probs(res.get_counts(i), 2 ** jobs[i]["n"], SHOTS)
                    for i in range(len(jobs))]

    for i, jb in enumerate(jobs):
        jb["fidelity"] = hellinger_fidelity(hw_dists[i], jb["ideal"])
        jb["tvd"] = tvd(hw_dists[i], jb["ideal"])
        jb["hw"] = hw_dists[i].tolist()
        jb["ideal"] = jb["ideal"].tolist()
        jb["x_sub"] = None; jb["U_A"] = None

    per_k = {}
    for k in K_LIST:
        fs = [jb["fidelity"] for jb in jobs if jb["k"] == k]
        gs = [jb["twoq"] for jb in jobs if jb["k"] == k]
        per_k[str(k)] = {"qubits": int(np.log2(k)), "mean_twoq": float(np.mean(gs)),
                         "mean_fidelity": float(np.mean(fs))}

    summary = {"backend": backend_name, "shots": SHOTS, "per_k": per_k, "runs": jobs}
    tag = "submit" if submit else "dryrun"
    with open(os.path.join(_EXPDIR, f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBackend: {backend_name}  (shots={SHOTS})")
    print(f"{'k':>4} {'qubits':>7} {'2q-gates':>9} {'Hellinger F':>12}")
    for k in K_LIST:
        pk = per_k[str(k)]
        print(f"{k:>4} {pk['qubits']:>7} {pk['mean_twoq']:>9.0f} {pk['mean_fidelity']:>12.4f}")
    return summary


# --------------------------------------------------------------------------
# Figure: fidelity vs k (left) + a representative k=8 distribution (right)
# --------------------------------------------------------------------------
def plot(summary=None, rep_k=8):
    summary = summary or load()
    plt = plotting.set_house_style()
    C_M, C_P, C_FLOOR = plotting.PALETTE["method"], plotting.PALETTE["global"], plotting.PALETTE["floor"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.3))

    ks = K_LIST
    fid = [summary["per_k"][str(k)]["mean_fidelity"] for k in ks]
    twoq = [summary["per_k"][str(k)]["mean_twoq"] for k in ks]
    qb = [summary["per_k"][str(k)]["qubits"] for k in ks]
    axL.plot(ks, fid, "o-", color=C_M, lw=2.0, ms=8, mec="white", mew=0.7)
    for k, f, g, q in zip(ks, fid, twoq, qb):
        axL.annotate(f"{q}q, {g:.0f} CX", xy=(k, f), xytext=(0, -14),
                     textcoords="offset points", ha="center", fontsize=8.5, color="0.3")
    axL.set_xscale("log", base=2)
    axL.set_xticks(ks); axL.set_xticklabels([str(k) for k in ks])
    axL.set_xlabel(r"subspace rank $k$"); axL.set_ylabel("Hellinger fidelity $F$")
    axL.set_title("(a) hardware fidelity vs subspace size")
    axL.grid(True, which="major", ls="-", lw=0.5, alpha=0.25)

    rep = next((r for r in summary["runs"] if r["k"] == rep_k), summary["runs"][0])
    hw = np.array(rep["hw"]); ideal = np.array(rep["ideal"])
    x = np.arange(len(hw)); w = 0.4
    axR.bar(x - w / 2, ideal, w, color=C_FLOOR, alpha=0.8, label="ideal")
    axR.bar(x + w / 2, hw, w, color=C_P, alpha=0.8, label="ibm\\_kobe")
    axR.set_xlabel("computational basis state")
    axR.set_ylabel("probability")
    axR.set_title(rf"(b) $k={rep['k']}$ ($F={rep['fidelity']:.2f}$)")
    axR.legend(fontsize=9.5, frameon=True)
    fig.tight_layout()
    plotting.savefig(fig, _FIG_STEM)
    plt.close(fig)


def load(tag="submit"):
    """Saved summary ('submit' = real device, 'dryrun' = Aer)."""
    path = os.path.join(_EXPDIR, f"{tag}.json")
    if not os.path.exists(path):                          # fall back to whichever exists
        path = os.path.join(_EXPDIR, "dryrun.json")
    with open(path) as f:
        return json.load(f)
