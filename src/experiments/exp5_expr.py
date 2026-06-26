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
on ibm_kobe (eff_dim 23.9 vs 17.2 at ~38 gates).

  from src.experiments.exp5_expr import simulate, plot, run_hardware, load
"""
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
SWEEP_L = {"method": [1, 2, 3, 4, 5], "pqc": [2, 4, 6, 8, 10, 12]}
N_DATA_EFFDIM = 1e5                         # data size in the effective-dimension formula

# Hardware sweep operating points: trace BOTH ansaetze across depth ON THE DEVICE
# so the measured method curve climbs THROUGH the PQC saturation ceiling, instead
# of sampling only the single matched point where the two curves coincide.
# method L=4 (~53 2q, sim eff_dim 34.8) sits well above the ~25 PQC ceiling, so it
# clears the ceiling even after the device noise penalty.  L=3 (method) / L=4 (pqc)
# remain the matched-resource pair, preserving the apples-to-apples +39% claim.
HW_SWEEP = {"method": [2, 3, 4], "pqc": [4, 8, 12]}
HW_EXTRA_SWEEP = {"method": [5]}          # supplementary point: method L=5
S_THETA_HW = 3
D_DATA_HW = 2

_SIM_JSON = "resource_sim.json"
_HW_JSON = "resource_hw.json"
_HW_JOBS_JSON = "resource_hw_jobs.json"
_HW_EXTRA_JOBS_JSON = "resource_hw_extra_jobs.json"
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


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def plot(res=None, hw=None, save_pdf=False):
    """Plot effective dimension vs two-qubit gate count (simulation + ibm_kobe)."""
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
        for kind, c in (("method", C_M), ("pqc", C_P)):
            pts = hw[kind] if isinstance(hw[kind], list) else [hw[kind]]
            tq = [r["twoq"] for r in pts]
            ed = [r["eff_dim"] for r in pts]
            ax.plot(tq, ed, "--", color=c, lw=1.4, alpha=0.85, zorder=5)
            ax.plot(tq, ed, "*", ms=11, color=c, mec="k", mew=0.55, zorder=6)
        hw_handle = mlines.Line2D(
            [], [], color="0.35", ls="--", lw=1.4,
            marker="*", ms=11, mec="k", mew=0.55,
            label=r"$\mathtt{ibm\_kobe}$ (DD+twirling)",
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + [hw_handle], labels + [r"$\mathtt{ibm\_kobe}$ (DD+twirling)"],
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
# Hardware: matched-2q effective dimension on ibm_kobe
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


def _hw_specs():
    """Fisher circuits for the full hardware depth sweep: per (kind, depth L,
    theta-sample, data) the unshifted circuit plus 2P parameter-shifted circuits.
    Deterministic (fixed seed) so the submit and fetch steps reconstruct the same
    circuit ordering without persisting every spec."""
    rng = np.random.default_rng(SEED + 21)
    X = make_data(D_DATA_HW, seed=SEED + 3)
    specs, circuits = [], []
    for kind, Ls in HW_SWEEP.items():
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


def submit_hardware(mitigation=False):
    """Transpile the full sweep and submit it to the device WITHOUT blocking.
    Job ids + chunk sizes + transpiled 2q counts are saved so `fetch_hardware`
    can retrieve and reduce the results once the queue clears.  mitigation=True
    enables device-level error suppression (DD + twirling) -- see
    hardware.submit_sampler."""
    specs, isa, twoq = dryrun()
    backend = hardware.get_backend(BACKEND_NAME)
    print(f"submitting {len(isa)} circuits to {backend.name} "
          f"(mitigation={'on' if mitigation else 'off'}) ...")
    job_ids, chunks = hardware.submit_sampler(isa, backend, shots=SHOTS, mitigation=mitigation)
    meta = {"backend": backend.name, "shots": SHOTS, "mitigation": mitigation,
            "job_ids": job_ids, "chunks": chunks, "n_circuits": len(isa),
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
    probs_map, gi = {}, 0                                   # flatten chunks back to submit order
    for res, n in zip(results, meta["chunks"]):
        for li in range(n):
            probs_map[specs[gi]] = hardware.counts_to_probs(
                hardware.get_counts(res, li), DIM, SHOTS)
            gi += 1
    twoq = {(k.split(":")[0], int(k.split(":")[1])): v for k, v in meta["twoq"].items()}
    hw = {"backend": meta["backend"], "shots": meta["shots"],
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
    print(f"saved -> result/exp5_expr/{_HW_JSON}")
    return hw


def run_hardware(mitigation=False):
    """Convenience for notebook use: submit then block until the jobs finish.
    For interactive/long queues prefer submit_hardware() + fetch_hardware()."""
    submit_hardware(mitigation=mitigation)
    return fetch_hardware()


# --------------------------------------------------------------------------
# Supplementary hardware point(s) (default: method L=5) -- appended to the
# existing resource_hw.json without re-running the full sweep.
# Uses an independent RNG seed (SEED+31) so thetas are fresh.
# --------------------------------------------------------------------------
def _hw_extra_specs(sweep=None):
    sweep = sweep or HW_EXTRA_SWEEP
    rng = np.random.default_rng(SEED + 31)
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


def submit_hardware_extra(sweep=None, mitigation=True):
    """Transpile and submit extra operating points to ibm_kobe (non-blocking).
    Job ids saved to result/exp5_expr/resource_hw_extra_jobs.json."""
    sweep = sweep or HW_EXTRA_SWEEP
    specs, circuits = _hw_extra_specs(sweep)
    pm, tag = hardware.make_passmanager()
    isa = [pm.run(c) for c in circuits]
    twoq_meta = {}
    for kind, Ls in sweep.items():
        for L in Ls:
            idx = [i for i, sp in enumerate(specs) if sp[0] == kind and sp[1] == L]
            twoq_val = int(np.median([isa[i].num_nonlocal_gates() for i in idx]))
            twoq_meta[f"{kind}:{L}"] = twoq_val
            print(f"  {kind} L={L} P={nparams(kind, L):3d} 2q={twoq_val}")
    backend = hardware.get_backend(BACKEND_NAME)
    print(f"submitting {len(isa)} circuits to {backend.name} "
          f"(mitigation={'on' if mitigation else 'off'}) ...")
    job_ids, chunks = hardware.submit_sampler(isa, backend, shots=SHOTS, mitigation=mitigation)
    meta = {"backend": backend.name, "shots": SHOTS, "mitigation": mitigation,
            "job_ids": job_ids, "chunks": chunks, "n_circuits": len(isa),
            "twoq": twoq_meta}
    with open(os.path.join(_EXPDIR, _HW_EXTRA_JOBS_JSON), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved {len(job_ids)} job id(s) -> result/exp5_expr/{_HW_EXTRA_JOBS_JSON}")
    return meta


def fetch_hardware_extra(sweep=None):
    """Retrieve the extra job, compute eff_dim, and append/overwrite in
    result/exp5_expr/resource_hw.json, then replot."""
    sweep = sweep or HW_EXTRA_SWEEP
    with open(os.path.join(_EXPDIR, _HW_EXTRA_JOBS_JSON)) as f:
        meta = json.load(f)
    specs, _ = _hw_extra_specs(sweep)
    assert len(specs) == meta["n_circuits"], "spec/circuit count drift"
    service = hardware.get_service()
    results = hardware.fetch_results(service, meta["job_ids"])
    probs_map, gi = {}, 0
    for res, n in zip(results, meta["chunks"]):
        for li in range(n):
            probs_map[specs[gi]] = hardware.counts_to_probs(
                hardware.get_counts(res, li), DIM, SHOTS)
            gi += 1
    twoq = {(k.split(":")[0], int(k.split(":")[1])): v
            for k, v in meta["twoq"].items()}
    hw_path = os.path.join(_EXPDIR, _HW_JSON)
    hw = json.load(open(hw_path))
    for kind, Ls in sweep.items():
        for L in Ls:
            P = nparams(kind, L)
            Fs = [_fisher_from_probs(probs_map, kind, L, s, list(range(D_DATA_HW)), P)
                  for s in range(S_THETA_HW)]
            rec = {"L": L, "P": P, "twoq": twoq[(kind, L)],
                   "eff_rank": float(np.mean([effective_rank(F) for F in Fs])),
                   "eff_dim": float(effective_dimension(Fs, N_DATA_EFFDIM))}
            print(f"[{meta['backend']}] {kind} L={L} P={P} 2q={rec['twoq']} "
                  f"eff_rank={rec['eff_rank']:.2f}  eff_dim={rec['eff_dim']:.2f}")
            existing = [r for r in hw.get(kind, []) if r["L"] != L]
            hw[kind] = sorted(existing + [rec], key=lambda r: r["L"])
    with open(hw_path, "w") as f:
        json.dump(hw, f, indent=2)
    print(f"appended -> result/exp5_expr/{_HW_JSON}")
    plot(*load())
    return hw


def load():
    """(sim, hw_or_None) from saved JSONs."""
    with open(os.path.join(_EXPDIR, _SIM_JSON)) as f:
        sim = json.load(f)
    hw_path = os.path.join(_EXPDIR, _HW_JSON)
    hw = json.load(open(hw_path)) if os.path.exists(hw_path) else None
    return sim, hw
