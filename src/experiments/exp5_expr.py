"""
Experiment V — Effective Dimension per Hardware Resource.

Reviewer's point: barren-plateau mitigation is only worthwhile if it does not
cost expressivity.  Raw Haar "expressibility" is the wrong metric -- by Holmes
et al. (2021) high Haar coverage co-occurs with barren plateaus.  We measure
USABLE capacity through the normalised effective dimension (Abbas et al. 2021)
and compare two 4-qubit ansaetze at matched TWO-QUBIT-GATE budget, sweeping depth:

  * METHOD   -- a trainable generator on every entangler: single-qubit rotations
                + parameterised couplings exp(-i theta_j G_j / 2) (rzz).
  * PLAIN PQC -- single-qubit ry interleaved with UNPARAMETERISED CZ.

Result: the plain ansatz saturates (effective-dimension ceiling ~25); the method
keeps climbing (>42).  At matched 2q cost the method wins by +41..77%, confirmed
on the selected IBM backend.

  from src.experiments.exp5_expr import simulate, plot, run_hardware, load
"""
import gzip
import os
import json

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from src import RESULT_DIR, SEED, SHOTS, BACKEND_NAME
from src.core import hardware
from src.analysis import plotting
from src.analysis.metrics import fisher_information, effective_rank, effective_dimension

_EXPDIR = os.path.join(RESULT_DIR, "exp5_expr")
os.makedirs(_EXPDIR, exist_ok=True)

N_QUBITS = 4
DIM = 2 ** N_QUBITS
RING = [(0, 1), (1, 2), (2, 3), (3, 0)]

D_DATA = 4                                  # input data points (fixed feature map)
S_THETA_SWEEP = 40                          # parameter samples for the Fisher average
SWEEP_L = {"method": [1, 2, 3, 4, 5, 6], "pqc": [2, 4, 6, 8, 10, 12]}
N_DATA_EFFDIM = 1e5                         # data size in the effective-dimension formula

# Hardware sweep operating points: trace BOTH ansaetze across depth ON THE DEVICE
# so the measured method curve climbs THROUGH the PQC saturation ceiling, instead
# of sampling only the single matched point where the two curves coincide.
# method L=4 (~53 2q, sim eff_dim 34.8) sits well above the ~25 PQC ceiling, so it
# clears the ceiling even after the device noise penalty.  L=3 (method) / L=4 (pqc)
# remain the matched-resource pair, preserving the apples-to-apples +39% claim.
HW_SWEEP = {"method": [2, 3, 4, 5, 6], "pqc": [4, 8, 12]}
HW_REMAINING_SWEEP = {"method": [1], "pqc": [2, 6, 10]}
HW_ADDITIONAL_SWEEP = {"method": [6]}
S_THETA_HW = 3
D_DATA_HW = 2

_SIM_JSON = "resource_sim.json"
_HW_JSON = "resource_hw.json"
_HW_JOBS_JSON = "resource_hw_jobs.json"
_RAW_JSON_GZ = "resource_raw_counts.json.gz"
_HW_REMAINING_JOBS_JSON = "resource_hw_remaining_jobs.json"
_RAW_REMAINING_JSON_GZ = "resource_raw_counts_remaining.json.gz"
_HW_ADDITIONAL_JOBS_JSON = "resource_hw_additional_jobs.json"
_RAW_ADDITIONAL_JSON_GZ = "resource_raw_counts_additional.json.gz"
_FIG_STEM = "exp5_effdim"


# --------------------------------------------------------------------------
# Data feature map (fixed, identical for both ansaetze) + depth-parameterised
# ansaetze.  The method spends each two-qubit gate on a parameterised rzz; the
# plain PQC spends it on an unparameterised CZ.
# --------------------------------------------------------------------------
def make_data(n, seed=SEED):
    return np.random.default_rng(seed + 777).uniform(0, np.pi, size=(n, N_QUBITS))


def _feature_map(qc, x):
    for q in range(N_QUBITS):
        qc.ry(float(x[q]), q)


def nparams(kind, L):
    return 12 * L if kind == "method" else 4 * (L + 1)


def build(kind, x, theta, L):
    qc = QuantumCircuit(N_QUBITS)
    _feature_map(qc, x)
    i = 0
    if kind == "method":
        for _ in range(L):
            for q in range(N_QUBITS):
                qc.ry(theta[i], q); i += 1
                qc.rz(theta[i], q); i += 1
            for (a, b) in RING:
                qc.rzz(theta[i], a, b); i += 1          # parameterised entangler
    else:
        for q in range(N_QUBITS):
            qc.ry(theta[i], q); i += 1
        for _ in range(L):
            for (a, b) in RING:
                qc.cz(a, b)                              # unparameterised entangler
            for q in range(N_QUBITS):
                qc.ry(theta[i], q); i += 1
    return qc


def _probs(kind, x, theta, L):
    return np.asarray(Statevector(build(kind, x, theta, L)).probabilities())


def fisher(kind, theta, X, L):
    """Classical Fisher averaged over the input set, via parameter-shift."""
    P = len(theta)
    F = np.zeros((P, P))
    for x in X:
        F += fisher_information(lambda th: _probs(kind, x, th, L), theta, P)
    return F / len(X)


# --------------------------------------------------------------------------
# Simulation: effective dimension vs transpiled two-qubit-gate count
# --------------------------------------------------------------------------
def _twoq_depth(pm, kind, L, X, rng):
    """Median transpiled 2q-gate count + depth over a few nonzero param sets
    (zero angles would let identity rotations and commuting CZ pairs cancel)."""
    P = nparams(kind, L)
    tq, dp = [], []
    for _ in range(3):
        isa = pm.run(build(kind, X[0], rng.uniform(0.3, 2 * np.pi - 0.3, P), L))
        tq.append(isa.num_nonlocal_gates()); dp.append(isa.depth())
    return int(np.median(tq)), int(np.median(dp))


def simulate():
    pm, tag = hardware.make_passmanager()
    rng = np.random.default_rng(SEED)
    X = make_data(D_DATA)
    res = {"backend": tag, "method": [], "pqc": []}
    for kind in ("method", "pqc"):
        for L in SWEEP_L[kind]:
            P = nparams(kind, L)
            twoq, depth = _twoq_depth(pm, kind, L, X, np.random.default_rng(123))
            Fs = [fisher(kind, rng.uniform(0, 2 * np.pi, P), X, L)
                  for _ in range(S_THETA_SWEEP)]
            Fbar = np.mean(Fs, axis=0)
            rec = {"L": L, "P": P, "twoq": twoq, "depth": depth,
                   "eff_rank": effective_rank(Fbar),
                   "eff_dim": effective_dimension(Fs, N_DATA_EFFDIM)}
            res[kind].append(rec)
            print(f"{kind:6s} L={L:2d} P={P:3d} 2q={twoq:3d} depth={depth:3d} "
                  f"eff_rank={rec['eff_rank']:6.2f}  eff_dim={rec['eff_dim']:6.2f}")
    with open(os.path.join(_EXPDIR, _SIM_JSON), "w") as f:
        json.dump(res, f, indent=2)
    return res


def refresh_sim_resources():
    """Refresh Kawasaki-dependent 2q/depth values without recomputing ideal Fishers."""
    with open(os.path.join(_EXPDIR, _SIM_JSON), encoding="utf-8") as file:
        result = json.load(file)
    passmanager, tag = hardware.make_passmanager()
    X = make_data(D_DATA)
    for kind in ("method", "pqc"):
        for record in result[kind]:
            twoq, depth = _twoq_depth(
                passmanager, kind, record["L"], X, np.random.default_rng(123)
            )
            record["twoq"] = twoq
            record["depth"] = depth
    result["backend"] = tag
    with open(os.path.join(_EXPDIR, _SIM_JSON), "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(f"refreshed simulation resource counts for {tag}")
    return result


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def plot(res=None, hw=None, save_pdf=False):
    """Plot effective dimension vs two-qubit gate count."""
    import matplotlib.lines as mlines
    if res is None:
        res, hw = load()
    plt = plotting.set_house_style()
    C_M, C_P = plotting.PALETTE["method"], plotting.PALETTE["global"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # ---- simulation curves (solid) ----
    for kind, c, mk, lab in (
        ("method", C_M, "o", "proposed method"),
        ("pqc",    C_P, "s", "plain PQC"),
    ):
        tq = [r["twoq"] for r in res[kind]]
        ed = [r["eff_dim"] for r in res[kind]]
        ax.plot(tq, ed, "-", color=c, lw=2.0)
        ax.plot(tq, ed, mk, color=c, ms=6.5, mec="white", mew=0.8, label=lab)

    # ---- PQC saturation ceiling ----
    ceil = res["pqc"][-1]["eff_dim"]
    ax.axhline(ceil, ls=(0, (5, 4)), lw=1.2, color=C_P, alpha=0.65)
    ax.text(17, ceil + 0.85, "PQC ceiling",
            color=C_P, fontsize=9, ha="left", style="italic", alpha=0.9)

    # ---- IBM hardware (dashed + star, colors match sim) ----
    if hw is not None:
        backend_label = hw.get("backend", BACKEND_NAME).replace("_", r"\_")
        for kind, c in (("method", C_M), ("pqc", C_P)):
            pts = hw[kind] if isinstance(hw[kind], list) else [hw[kind]]
            tq = [r["twoq"] for r in pts]
            ed = [r["eff_dim"] for r in pts]
            ax.plot(tq, ed, "--", color=c, lw=1.4, alpha=0.85, zorder=5)
            ax.plot(tq, ed, "*", ms=11, color=c, mec="k", mew=0.55, zorder=6)
        hw_handle = mlines.Line2D(
            [], [], color="0.35", ls="--", lw=1.4,
            marker="*", ms=11, mec="k", mew=0.55,
            label=rf"$\mathtt{{{backend_label}}}$ (DD+twirling)",
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + [hw_handle],
                  labels + [rf"$\mathtt{{{backend_label}}}$ (DD+twirling)"],
                  fontsize=9.5, loc="upper left", frameon=True)
    else:
        ax.legend(fontsize=9.5, loc="upper left", frameon=True)

    ax.set_xlabel("transpiled two-qubit gates")
    ax.set_ylabel("effective dimension")
    ax.grid(True, which="major", ls="-", lw=0.4, alpha=0.2)
    fig.tight_layout()
    plotting.savefig(fig, _FIG_STEM, pdf=save_pdf)
    plt.close(fig)


# --------------------------------------------------------------------------
# Hardware: matched-2q effective dimension on the selected IBM backend
# --------------------------------------------------------------------------
def _fisher_from_probs(probs_map, kind, L, s, d_list, P):
    F = np.zeros((P, P))
    for d in d_list:
        p = probs_map[(kind, L, s, d, -1, 0)]
        dp = np.zeros((P, len(p)))
        for j in range(P):
            dp[j] = 0.5 * (probs_map[(kind, L, s, d, j, +1)] - probs_map[(kind, L, s, d, j, -1)])
        mask = p > 1e-9
        dlog = dp[:, mask] / p[mask]
        F += (dlog * p[mask]) @ dlog.T
    return F / len(d_list)


def _build_hw_specs(sweep, rng_seed):
    """Fisher circuits for the full hardware depth sweep: per (kind, depth L,
    theta-sample, data) the unshifted circuit plus 2P parameter-shifted circuits.
    Deterministic (fixed seed) so the submit and fetch steps reconstruct the same
    circuit ordering without persisting every spec."""
    rng = np.random.default_rng(rng_seed)
    X = make_data(D_DATA_HW, seed=SEED + 3)
    specs, circuits = [], []
    for kind, Ls in sweep.items():
        for L in Ls:
            P = nparams(kind, L)
            for s in range(S_THETA_HW):
                theta = rng.uniform(0, 2 * np.pi, P)
                for d in range(D_DATA_HW):
                    qc = build(kind, X[d], theta, L); qc.measure_all()
                    circuits.append(qc); specs.append((kind, L, s, d, -1, 0))
                    for j in range(P):
                        for sign in (+1, -1):
                            tt = theta.copy(); tt[j] += sign * np.pi / 2
                            qc = build(kind, X[d], tt, L); qc.measure_all()
                            circuits.append(qc); specs.append((kind, L, s, d, j, sign))
    return specs, circuits


def _hw_specs():
    return _build_hw_specs(HW_SWEEP, SEED + 21)


def dryrun():
    specs, circuits = _hw_specs()
    pm, tag = hardware.make_passmanager()
    isa = [pm.run(c) for c in circuits]
    print(f"hardware depth sweep ({tag}):")
    twoq = {}
    for kind, Ls in HW_SWEEP.items():
        for L in Ls:
            idx = [i for i, sp in enumerate(specs) if sp[0] == kind and sp[1] == L]
            twoq[(kind, L)] = int(np.median([isa[i].num_nonlocal_gates() for i in idx]))
            print(f"  {kind:6s} L={L:2d} P={nparams(kind, L):3d} circuits={len(idx):4d} "
                  f"median 2q={twoq[(kind, L)]}")
    print(f"total circuits={len(circuits)} x {SHOTS} shots = {len(circuits)*SHOTS:,} shots")
    return specs, isa, twoq


def submit_hardware(mitigation=True):
    """Transpile the full sweep and submit it to the device WITHOUT blocking.
    Job ids + chunk sizes + transpiled 2q counts are saved so `fetch_hardware`
    can retrieve and reduce the results once the queue clears.  mitigation=True
    enables device-level error suppression (DD + twirling) -- see
    hardware.submit_sampler."""
    specs, isa, twoq = dryrun()
    backend = hardware.get_backend(BACKEND_NAME)
    print(f"submitting {len(isa)} circuits to {backend.name} "
          f"(mitigation={'on' if mitigation else 'off'}) ...")
    batch_id, job_ids, chunks = hardware.submit_sampler(
        isa, backend, shots=SHOTS, mitigation=mitigation
    )
    meta = {**hardware.backend_metadata(backend),
            "experiment": "exp5_expr", "batch_id": batch_id,
            "mitigation": mitigation,
            "job_ids": job_ids, "chunks": chunks, "n_circuits": len(isa),
            "sweep": HW_SWEEP, "s_theta_hw": S_THETA_HW,
            "d_data_hw": D_DATA_HW,
            "twoq": {f"{k}:{L}": v for (k, L), v in twoq.items()}}
    with open(os.path.join(_EXPDIR, _HW_JOBS_JSON), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved {len(job_ids)} job id(s) -> result/exp5_expr/{_HW_JOBS_JSON}")
    return meta


def fetch_hardware():
    """Retrieve the submitted sweep, reduce to effective dimension per operating
    point, and write result/exp5_expr/resource_hw.json."""
    with open(os.path.join(_EXPDIR, _HW_JOBS_JSON)) as f:
        meta = json.load(f)
    specs, _ = _hw_specs()
    assert len(specs) == meta["n_circuits"], "spec/circuit count drift -- did the sweep config change?"
    service = hardware.get_service()
    results = hardware.fetch_results(service, meta["job_ids"])
    probs_map, raw_counts, gi = {}, [], 0
    for res, n in zip(results, meta["chunks"]):
        for li in range(n):
            counts = hardware.get_counts(res, li)
            raw_counts.append(counts)
            probs_map[specs[gi]] = hardware.counts_to_probs(counts, DIM)
            gi += 1
    twoq = {(k.split(":")[0], int(k.split(":")[1])): v for k, v in meta["twoq"].items()}
    hw = {"backend": meta["backend"], "shots": meta["shots"],
          "batch_id": meta["batch_id"], "job_ids": meta["job_ids"],
          "submitted_at_utc": meta["submitted_at_utc"],
          "calibration_last_update": meta["calibration_last_update"],
          "qiskit_version": meta["qiskit_version"],
          "qiskit_ibm_runtime_version": meta["qiskit_ibm_runtime_version"],
          "mitigation": meta.get("mitigation", False), "method": [], "pqc": []}
    for kind, Ls in HW_SWEEP.items():
        for L in Ls:
            P = nparams(kind, L)
            Fs = [_fisher_from_probs(probs_map, kind, L, s, list(range(D_DATA_HW)), P)
                  for s in range(S_THETA_HW)]
            rec = {"L": L, "P": P, "twoq": twoq[(kind, L)],
                   "eff_rank": float(np.mean([effective_rank(F) for F in Fs])),
                   "eff_dim": float(effective_dimension(Fs, N_DATA_EFFDIM))}
            hw[kind].append(rec)
            print(f"[{meta['backend']}] {kind:6s} L={L:2d} P={P:3d} 2q={rec['twoq']:3d} "
                  f"eff_rank={rec['eff_rank']:.2f}  eff_dim={rec['eff_dim']:.2f}")
    with open(os.path.join(_EXPDIR, _HW_JSON), "w") as f:
        json.dump(hw, f, indent=2)
    with gzip.open(os.path.join(_EXPDIR, _RAW_JSON_GZ), "wt", encoding="utf-8") as file:
        json.dump({"job_ids": meta["job_ids"], "counts": raw_counts}, file)
    print(f"saved -> result/exp5_expr/{_HW_JSON}")
    print(f"saved compressed raw counts -> result/exp5_expr/{_RAW_JSON_GZ}")
    return hw


def run_hardware(mitigation=True):
    """Convenience for notebook use: submit then block until the jobs finish.
    For interactive/long queues prefer submit_hardware() + fetch_hardware()."""
    submit_hardware(mitigation=mitigation)
    return fetch_hardware()


def _remaining_hw_specs():
    return _build_hw_specs(HW_REMAINING_SWEEP, SEED + 31)


def dryrun_remaining():
    specs, circuits = _remaining_hw_specs()
    passmanager, tag = hardware.make_passmanager()
    isa = [passmanager.run(circuit) for circuit in circuits]
    twoq = {}
    print(f"remaining hardware points ({tag}):")
    for kind, depths in HW_REMAINING_SWEEP.items():
        for depth in depths:
            indices = [
                i for i, spec in enumerate(specs)
                if spec[0] == kind and spec[1] == depth
            ]
            twoq[(kind, depth)] = int(np.median([
                isa[i].num_nonlocal_gates() for i in indices
            ]))
            print(
                f"  {kind:6s} L={depth:2d} P={nparams(kind, depth):3d} "
                f"circuits={len(indices):4d} median 2q={twoq[(kind, depth)]}"
            )
    print(
        f"total remaining circuits={len(circuits)} x {SHOTS} shots "
        f"= {len(circuits) * SHOTS:,} shots"
    )
    return specs, isa, twoq


def submit_hardware_remaining(mitigation=True):
    """Submit the four points omitted from the initial seven-point sweep."""
    _, isa, twoq = dryrun_remaining()
    backend = hardware.get_backend(BACKEND_NAME)
    print(
        f"submitting {len(isa)} remaining circuits to {backend.name} "
        f"(mitigation={'on' if mitigation else 'off'}) ..."
    )
    batch_id, job_ids, chunks = hardware.submit_sampler(
        isa, backend, shots=SHOTS, mitigation=mitigation
    )
    meta = {
        **hardware.backend_metadata(backend),
        "experiment": "exp5_expr_remaining",
        "batch_id": batch_id,
        "mitigation": mitigation,
        "job_ids": job_ids,
        "chunks": chunks,
        "n_circuits": len(isa),
        "sweep": HW_REMAINING_SWEEP,
        "rng_seed": SEED + 31,
        "s_theta_hw": S_THETA_HW,
        "d_data_hw": D_DATA_HW,
        "twoq": {f"{kind}:{depth}": value
                 for (kind, depth), value in twoq.items()},
    }
    path = os.path.join(_EXPDIR, _HW_REMAINING_JOBS_JSON)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
    print(f"saved {len(job_ids)} job id(s) -> {path}")
    return meta


def fetch_hardware_remaining():
    """Fetch and merge the remaining four points into resource_hw.json."""
    jobs_path = os.path.join(_EXPDIR, _HW_REMAINING_JOBS_JSON)
    with open(jobs_path, encoding="utf-8") as file:
        meta = json.load(file)
    specs, _ = _remaining_hw_specs()
    assert len(specs) == meta["n_circuits"], "remaining spec/circuit count drift"
    service = hardware.get_service()
    results = hardware.fetch_results(service, meta["job_ids"])
    probs_map, raw_counts, global_index = {}, [], 0
    for result, chunk_size in zip(results, meta["chunks"]):
        for local_index in range(chunk_size):
            counts = hardware.get_counts(result, local_index)
            raw_counts.append(counts)
            probs_map[specs[global_index]] = hardware.counts_to_probs(counts, DIM)
            global_index += 1
    assert global_index == meta["n_circuits"]

    twoq = {
        (key.split(":")[0], int(key.split(":")[1])): value
        for key, value in meta["twoq"].items()
    }
    hw_path = os.path.join(_EXPDIR, _HW_JSON)
    with open(hw_path, encoding="utf-8") as file:
        hw = json.load(file)
    for kind, depths in HW_REMAINING_SWEEP.items():
        for depth in depths:
            parameter_count = nparams(kind, depth)
            fishers = [
                _fisher_from_probs(
                    probs_map, kind, depth, sample,
                    list(range(D_DATA_HW)), parameter_count
                )
                for sample in range(S_THETA_HW)
            ]
            record = {
                "L": depth,
                "P": parameter_count,
                "twoq": twoq[(kind, depth)],
                "eff_rank": float(np.mean([
                    effective_rank(fisher_matrix) for fisher_matrix in fishers
                ])),
                "eff_dim": float(effective_dimension(fishers, N_DATA_EFFDIM)),
            }
            existing = [item for item in hw[kind] if item["L"] != depth]
            hw[kind] = sorted(existing + [record], key=lambda item: item["L"])
            print(
                f"[{meta['backend']}] {kind:6s} L={depth:2d} "
                f"P={parameter_count:3d} 2q={record['twoq']:3d} "
                f"eff_rank={record['eff_rank']:.2f} "
                f"eff_dim={record['eff_dim']:.2f}"
            )
    hw.setdefault("batches", [{
        "batch_id": hw.get("batch_id"),
        "job_ids": list(hw.get("job_ids", [])),
        "submitted_at_utc": hw.get("submitted_at_utc"),
    }])
    hw["batches"].append({
        "batch_id": meta["batch_id"],
        "job_ids": meta["job_ids"],
        "submitted_at_utc": meta["submitted_at_utc"],
    })
    hw["job_ids"] = hw.get("job_ids", []) + meta["job_ids"]
    with open(hw_path, "w", encoding="utf-8") as file:
        json.dump(hw, file, indent=2)
    raw_path = os.path.join(_EXPDIR, _RAW_REMAINING_JSON_GZ)
    with gzip.open(raw_path, "wt", encoding="utf-8") as file:
        json.dump({"job_ids": meta["job_ids"], "counts": raw_counts}, file)
    print(f"merged remaining points -> {hw_path}")
    print(f"saved compressed remaining raw counts -> {raw_path}")
    return hw


def submit_hardware_additional(mitigation=True):
    """Submit only the newly added method L=6 operating point."""
    specs, circuits = _build_hw_specs(HW_ADDITIONAL_SWEEP, SEED + 41)
    passmanager, tag = hardware.make_passmanager()
    isa = [passmanager.run(circuit) for circuit in circuits]
    twoq = int(np.median([circuit.num_nonlocal_gates() for circuit in isa]))
    backend = hardware.get_backend(BACKEND_NAME)
    print(f"submitting {len(isa)} additional circuits to {backend.name} "
          f"(method L=6, median 2q={twoq}, mitigation={'on' if mitigation else 'off'})")
    batch_id, job_ids, chunks = hardware.submit_sampler(
        isa, backend, shots=SHOTS, mitigation=mitigation
    )
    meta = {
        **hardware.backend_metadata(backend), "experiment": "exp5_expr_additional",
        "batch_id": batch_id, "job_ids": job_ids, "chunks": chunks,
        "n_circuits": len(isa), "mitigation": mitigation,
        "sweep": HW_ADDITIONAL_SWEEP, "rng_seed": SEED + 41,
        "twoq": {"method:6": twoq},
    }
    with open(os.path.join(_EXPDIR, _HW_ADDITIONAL_JOBS_JSON), "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
    return meta


def fetch_hardware_additional():
    """Fetch and merge only the newly added method L=6 point."""
    with open(os.path.join(_EXPDIR, _HW_ADDITIONAL_JOBS_JSON), encoding="utf-8") as file:
        meta = json.load(file)
    specs, _ = _build_hw_specs(HW_ADDITIONAL_SWEEP, meta["rng_seed"])
    service = hardware.get_service()
    results = hardware.fetch_results(service, meta["job_ids"])
    probs_map, raw_counts, index = {}, [], 0
    for result, chunk_size in zip(results, meta["chunks"]):
        for local_index in range(chunk_size):
            counts = hardware.get_counts(result, local_index)
            raw_counts.append(counts)
            probs_map[specs[index]] = hardware.counts_to_probs(counts, DIM)
            index += 1
    assert index == meta["n_circuits"]
    parameter_count = nparams("method", 6)
    fishers = [_fisher_from_probs(probs_map, "method", 6, sample,
                                  list(range(D_DATA_HW)), parameter_count)
               for sample in range(S_THETA_HW)]
    record = {
        "L": 6, "P": parameter_count, "twoq": meta["twoq"]["method:6"],
        "eff_rank": float(np.mean([effective_rank(fisher) for fisher in fishers])),
        "eff_dim": float(effective_dimension(fishers, N_DATA_EFFDIM)),
    }
    hw_path = os.path.join(_EXPDIR, _HW_JSON)
    with open(hw_path, encoding="utf-8") as file:
        hw = json.load(file)
    assert hw["backend"] == meta["backend"], "cannot merge different backends"
    hw["method"] = sorted([item for item in hw["method"] if item["L"] != 6] + [record],
                          key=lambda item: item["L"])
    hw.setdefault("batches", [{
        "batch_id": hw.get("batch_id"), "job_ids": list(hw.get("job_ids", [])),
        "submitted_at_utc": hw.get("submitted_at_utc"),
    }])
    hw["batches"].append({
        "batch_id": meta["batch_id"], "job_ids": meta["job_ids"],
        "submitted_at_utc": meta["submitted_at_utc"],
    })
    hw["job_ids"] += meta["job_ids"]
    with open(hw_path, "w", encoding="utf-8") as file:
        json.dump(hw, file, indent=2)
    with gzip.open(os.path.join(_EXPDIR, _RAW_ADDITIONAL_JSON_GZ), "wt", encoding="utf-8") as file:
        json.dump({"job_ids": meta["job_ids"], "counts": raw_counts}, file)
    plot(*load(), save_pdf=True)
    return record


def load():
    """(sim, hw_or_None) from saved JSONs."""
    with open(os.path.join(_EXPDIR, _SIM_JSON)) as f:
        sim = json.load(f)
    hw_path = os.path.join(_EXPDIR, _HW_JSON)
    hw = json.load(open(hw_path)) if os.path.exists(hw_path) else None
    return sim, hw
