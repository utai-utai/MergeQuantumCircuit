"""
Experiment III — zero-shot subspace-circuit fidelity on IBM hardware.

The experiment trains the same deterministic MNIST residual core used by the
original run, compiles V = U_A S for k in {2, 4, 8, 16}, and compares measured
distributions with ideal statevector distributions. Hardware work is split into
submit/fetch steps and submitted through an IBM Runtime Batch.
"""
import gzip
import json
import os

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

from src import BACKEND_NAME, RESULT_DIR, SEED, SHOTS
from src.analysis import plotting
from src.analysis.metrics import hellinger_fidelity, tvd
from src.core import hardware
from src.core.data import mnist_split_loaders
from src.core.geometric_qml import transfer_map
from src.experiments.validation import ClassicalResNet, DEVICE, train_model

_EXPDIR = os.path.join(RESULT_DIR, "exp3_hardware")
os.makedirs(_EXPDIR, exist_ok=True)

K_LIST = [2, 4, 8, 16]
N_INPUTS = 2
_JOBS_JSON = os.path.join(_EXPDIR, "hw_jobs.json")
_HW_JSON = os.path.join(_EXPDIR, "hw.json")
_RAW_JSON_GZ = os.path.join(_EXPDIR, "raw_counts.json.gz")
_FIG_STEM = "exp3_fidelity"


def _clean_unitary(matrix):
    x, _, vh = np.linalg.svd(matrix.astype(np.complex128))
    return x @ vh


def _state_prep_unitary(state):
    """Return a unitary S whose first column is the normalised input state."""
    dim = len(state)
    cols = [state.astype(np.complex128) / np.linalg.norm(state)]
    for basis in np.eye(dim, dtype=np.complex128):
        vec = basis.copy()
        for col in cols:
            vec -= (np.conj(col) @ vec) * col
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            cols.append(vec / norm)
        if len(cols) == dim:
            break
    return np.column_stack(cols)


def make_circuit(state, unitary):
    n_qubits = int(np.log2(len(state)))
    compiled = _clean_unitary(
        unitary.astype(np.complex128) @ _state_prep_unitary(state)
    )
    circuit = QuantumCircuit(n_qubits)
    circuit.append(UnitaryGate(compiled), range(n_qubits))
    return circuit


def _prepare_circuits():
    """Build the eight deterministic circuits and their ideal distributions."""
    torch.manual_seed(SEED)
    train_all, _, _, test = mnist_split_loaders()
    model = train_model(
        ClassicalResNet(hidden_dim=64).to(DEVICE), train_all, epochs=2
    )
    weight = model.core_layer.weight.detach().cpu()

    images, _ = next(iter(test))
    with torch.no_grad():
        hidden = model.relu(
            model.fc_in(model.flatten(images[:N_INPUTS].to(DEVICE)))
        ).to(torch.complex64).cpu()

    records, circuits = [], []
    for k in K_LIST:
        frame, subspace_u, _ = transfer_map(weight, k)
        subspace_u = _clean_unitary(subspace_u.numpy())
        states = (hidden @ frame.conj()).numpy()
        states /= np.linalg.norm(states, axis=1, keepdims=True) + 1e-12
        for input_idx, state in enumerate(states):
            circuit = make_circuit(state, subspace_u)
            ideal = np.asarray(Statevector(circuit).probabilities())
            circuit.measure_all()
            circuits.append(circuit)
            records.append({
                "k": k,
                "qubits": int(np.log2(k)),
                "input_idx": input_idx,
                "ideal": ideal.tolist(),
            })
    return records, circuits


def dryrun():
    records, circuits = _prepare_circuits()
    passmanager, tag = hardware.make_passmanager()
    isa = [passmanager.run(circuit) for circuit in circuits]
    for record, circuit in zip(records, isa):
        record["twoq"] = circuit.num_nonlocal_gates()
        record["depth"] = circuit.depth()
    print(f"Exp III dry-run on {tag}: {len(isa)} circuits x {SHOTS} shots")
    for record in records:
        print(
            f"  k={record['k']:2d} input={record['input_idx']} "
            f"2q={record['twoq']:3d} depth={record['depth']:4d}"
        )
    return records, isa


def submit_hardware():
    records, isa = dryrun()
    backend = hardware.get_backend(BACKEND_NAME)
    batch_id, job_ids, chunks = hardware.submit_sampler(
        isa, backend, shots=SHOTS, mitigation=False
    )
    meta = {
        **hardware.backend_metadata(backend),
        "experiment": "exp3_hardware",
        "batch_id": batch_id,
        "job_ids": job_ids,
        "chunks": chunks,
        "n_circuits": len(isa),
        "k_list": K_LIST,
        "n_inputs": N_INPUTS,
        "records": records,
    }
    with open(_JOBS_JSON, "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
    print(f"saved submission metadata -> {_JOBS_JSON}")
    return meta


def fetch_hardware():
    with open(_JOBS_JSON, encoding="utf-8") as file:
        meta = json.load(file)
    service = hardware.get_service()
    results = hardware.fetch_results(service, meta["job_ids"])

    counts, flat_index = [], 0
    for result, chunk_size in zip(results, meta["chunks"]):
        for local_index in range(chunk_size):
            counts.append(hardware.get_counts(result, local_index))
            flat_index += 1
    assert flat_index == meta["n_circuits"]

    runs = []
    for record, raw in zip(meta["records"], counts):
        measured = hardware.counts_to_probs(raw, 2 ** record["qubits"])
        ideal = np.asarray(record["ideal"])
        runs.append({
            **record,
            "fidelity": hellinger_fidelity(measured, ideal),
            "tvd": tvd(measured, ideal),
            "measured": measured.tolist(),
        })

    per_k = {}
    for k in K_LIST:
        selected = [run for run in runs if run["k"] == k]
        per_k[str(k)] = {
            "qubits": int(np.log2(k)),
            "mean_twoq": float(np.mean([run["twoq"] for run in selected])),
            "mean_depth": float(np.mean([run["depth"] for run in selected])),
            "mean_fidelity": float(np.mean([run["fidelity"] for run in selected])),
            "std_fidelity": float(np.std([run["fidelity"] for run in selected])),
            "mean_tvd": float(np.mean([run["tvd"] for run in selected])),
        }

    summary = {
        key: value for key, value in meta.items()
        if key not in {"records", "job_ids"}
    }
    summary["job_ids"] = meta["job_ids"]
    summary["per_k"] = per_k
    summary["runs"] = runs
    with open(_HW_JSON, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    with gzip.open(_RAW_JSON_GZ, "wt", encoding="utf-8") as file:
        json.dump({"job_ids": meta["job_ids"], "counts": counts}, file)

    print(f"saved hardware results -> {_HW_JSON}")
    print(f"saved compressed raw counts -> {_RAW_JSON_GZ}")
    return summary


def load():
    with open(_HW_JSON, encoding="utf-8") as file:
        return json.load(file)


def plot(summary=None, save_pdf=False):
    summary = summary or load()
    plt = plotting.set_house_style()
    color = plotting.PALETTE["method"]
    fig, ax = plt.subplots(figsize=(5.5, 4.3))

    ks = K_LIST
    fidelity = [summary["per_k"][str(k)]["mean_fidelity"] for k in ks]
    twoq = [summary["per_k"][str(k)]["mean_twoq"] for k in ks]
    qubits = [summary["per_k"][str(k)]["qubits"] for k in ks]
    ax.plot(ks, fidelity, "o-", color=color, lw=2.0, ms=8, mec="white", mew=0.7)
    offsets = {
        2: (0, -10, "left", "top"),
        4: (0, -10, "center", "top"),
        8: (65, 8, "right", "top"),
        16: (-12, 0, "right", "bottom"),
    }
    for k, value, gates, n_qubits in zip(ks, fidelity, twoq, qubits):
        dx, dy, ha, va = offsets[k]
        ax.annotate(
            rf"$n={n_qubits}$, $N_{{2q}}={gates:.0f}$",
            xy=(k, value),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=8.5,
            color="0.3",
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel(r"subspace rank $k$")
    ax.set_ylabel("Hellinger fidelity $F$")
    ax.grid(True, which="major", ls="-", lw=0.5, alpha=0.25)
    fig.tight_layout()
    plotting.savefig(fig, _FIG_STEM, pdf=save_pdf)
    plt.close(fig)
