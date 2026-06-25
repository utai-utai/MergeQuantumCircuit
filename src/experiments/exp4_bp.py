"""
Experiment IV — Trainability from 16 to 128 qubits (barren plateaus, Theorem 2).

Compares the gradient variance of a GLOBAL hardware-efficient ansatz on n qubits
(Var ~ 2^-n, barren plateau) against our SUBSPACE-restricted ansatz whose active
dynamics occupy only m = log2(k) qubits inside the n-qubit register (Var
independent of n, ~ Omega(1/k)).

Observable: parity Z^{otimes a} over the qubits the ansatz actually drives
(a = n global, a = m subspace).  The n - m idle subspace qubits stay in |0>
(deterministic +1 parity) and are excluded from the hardware observable, so the
trainability claim is isolated from the exponential readout-error suppression a
global Z^{otimes n} would incur on the idle qubits.

  from src.experiments.exp4_bp import simulate, run_hardware, plot, load
"""
import os
import json

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from src import RESULT_DIR, SHOTS, BACKEND_NAME
from src.core import hardware
from src.analysis import plotting
from src.analysis.metrics import parity_op

_EXPDIR = os.path.join(RESULT_DIR, "exp4_bp")
os.makedirs(_EXPDIR, exist_ok=True)

K = 16                       # subspace dim -> m = 4 active qubits
M = int(np.log2(K))
GLOBAL_LAYERS = 4
SUB_LAYERS = 2
N_LIST_SIM = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]            # exact statevector (global)
N_LIST_SUB_SIM = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 40, 80, 128]
N_LIST_HW = [16, 32, 64, 96, 128]
S_SIM = 80                   # random inits for simulation variance
S_HW = 20                    # random inits per (n, kind) for hardware variance
RNG = np.random.default_rng(0)

_SIM_JSON = "sim.json"
_HW_JSON = "hw.json"
_FIG_STEM = "exp4_gradient_variance"


# --------------------------------------------------------------------------
# Ansaetze
# --------------------------------------------------------------------------
def global_ansatz(n, params):
    p = params.reshape(GLOBAL_LAYERS, n)
    qc = QuantumCircuit(n)
    for l in range(GLOBAL_LAYERS):
        for q in range(n):
            qc.ry(p[l, q], q)
        for q in range(n - 1):
            qc.cz(q, q + 1)
    return qc


def subspace_ansatz(width, params):
    """Subspace ansatz on `width` qubits; only the first m carry dynamics."""
    m = min(M, width)
    p = params.reshape(SUB_LAYERS, m)
    qc = QuantumCircuit(width)
    for l in range(SUB_LAYERS):
        for q in range(m):
            qc.ry(p[l, q], q)
        for q in range(m - 1):
            qc.cz(q, q + 1)
    return qc


def active_count(kind, n):
    return n if kind == "global" else min(M, n)


def n_params(kind, n):
    return GLOBAL_LAYERS * n if kind == "global" else SUB_LAYERS * min(M, n)


# --------------------------------------------------------------------------
# Simulation: exact parity expectation + parameter-shift gradient variance
# --------------------------------------------------------------------------
def _sim_circuit(kind, n, params):
    if kind == "global":
        return global_ansatz(n, params)
    return subspace_ansatz(min(M, n), params)    # compact: m qubits only


def cost_sim(kind, n, params, op):
    return float(np.real(Statevector(_sim_circuit(kind, n, params)).expectation_value(op)))


def grad0_sim(kind, n, params, op):
    pp = params.copy(); pp[0] += np.pi / 2
    pm = params.copy(); pm[0] -= np.pi / 2
    return 0.5 * (cost_sim(kind, n, pp, op) - cost_sim(kind, n, pm, op))


def _var_sim(kind, n):
    op = parity_op(active_count(kind, n))
    npar = n_params(kind, n)
    grads = [grad0_sim(kind, n, RNG.uniform(0, 2 * np.pi, npar), op) for _ in range(S_SIM)]
    return float(np.var(grads))


def simulate():
    out = {"n_global": N_LIST_SIM, "n_subspace": N_LIST_SUB_SIM,
           "global_var": [], "subspace_var": []}
    for n in N_LIST_SIM:
        out["global_var"].append(_var_sim("global", n))
        print(f"[sim] n={n:3d} global   Var={out['global_var'][-1]:.3e}")
    for n in N_LIST_SUB_SIM:
        out["subspace_var"].append(_var_sim("subspace", n))
        print(f"[sim] n={n:3d} subspace Var={out['subspace_var'][-1]:.3e}")
    with open(os.path.join(_EXPDIR, _SIM_JSON), "w") as f:
        json.dump(out, f, indent=2)
    return out


# --------------------------------------------------------------------------
# Hardware circuits (measure only the active sub-register)
# --------------------------------------------------------------------------
def make_hw(kind, n, params):
    a = active_count(kind, n)
    qc = global_ansatz(n, params) if kind == "global" else subspace_ansatz(n, params)
    meas = QuantumCircuit(n, a)
    meas.compose(qc, inplace=True)
    meas.measure(range(a), range(a))             # idle qubits excluded from observable
    return meas


def _build_specs():
    specs, circuits = [], []
    for n in N_LIST_HW:
        for kind in ("global", "subspace"):
            npar = n_params(kind, n)
            for s in range(S_HW):
                params = RNG.uniform(0, 2 * np.pi, npar)
                for sign in (+1, -1):
                    pp = params.copy(); pp[0] += sign * np.pi / 2
                    circuits.append(make_hw(kind, n, pp))
                    specs.append((n, kind, s, sign))
    return specs, circuits


def _gate_table(specs, isa):
    table = {}
    for n in N_LIST_HW:
        for kind in ("global", "subspace"):
            idx = [i for i, sp in enumerate(specs) if sp[0] == n and sp[1] == kind]
            twoq = [isa[i].num_nonlocal_gates() for i in idx]
            depth = [isa[i].depth() for i in idx]
            table[(n, kind)] = {"twoq": int(np.median(twoq)), "depth": int(np.median(depth))}
    return table


def dryrun():
    """Build + transpile the full sweep without device time; predict subspace
    variance exactly and report gate counts."""
    specs, circuits = _build_specs()
    pm, _ = hardware.make_passmanager()
    print(f"transpiling {len(circuits)} circuits ...")
    isa = [pm.run(c) for c in circuits]
    table = _gate_table(specs, isa)
    print(f"\n{'n':>4} {'kind':>9} {'2q-gates':>9} {'depth':>7}")
    for n in N_LIST_HW:
        for kind in ("global", "subspace"):
            t = table[(n, kind)]
            print(f"{n:>4} {kind:>9} {t['twoq']:>9} {t['depth']:>7}")
    pred = {n: _var_sim("subspace", n) for n in N_LIST_HW}
    print("\npredicted subspace Var (exact sim):")
    for n in N_LIST_HW:
        print(f"  n={n:>4}: {pred[n]:.3e}")
    return specs, isa, table


def run_hardware():
    specs, circuits = _build_specs()
    pm, _ = hardware.make_passmanager()
    backend = hardware.get_backend(BACKEND_NAME)
    print(f"transpiling {len(circuits)} circuits for {backend.name} ...")
    isa = [pm.run(c) for c in circuits]
    table = _gate_table(specs, isa)
    print(f"submitting {len(isa)} circuits x {SHOTS} shots to {backend.name}")
    res = hardware.run_sampler(isa, backend, shots=SHOTS)

    parity = [hardware.counts_to_parity(hardware.get_counts(res, i, creg="c"), SHOTS)
              for i in range(len(specs))]
    grads = {n: {"global": {}, "subspace": {}} for n in N_LIST_HW}
    for (n, kind, s, sign), val in zip(specs, parity):
        grads[n][kind].setdefault(s, {})[sign] = val

    result = {"backend": backend.name, "n": N_LIST_HW, "shots": SHOTS,
              "shot_floor_var": 1.0 / SHOTS,
              "gate_table": {f"{n}_{kind}": table[(n, kind)]
                             for n in N_LIST_HW for kind in ("global", "subspace")},
              "global_var": [], "subspace_var": [], "detail": {}}
    for n in N_LIST_HW:
        for kind in ("global", "subspace"):
            gs = [0.5 * (grads[n][kind][s][+1] - grads[n][kind][s][-1])
                  for s in grads[n][kind]]
            result["detail"][f"{n}_{kind}"] = gs
            result[f"{kind}_var"].append(float(np.var(gs)))
        print(f"[{backend.name} n={n:>4}] global Var={result['global_var'][-1]:.3e} | "
              f"subspace Var={result['subspace_var'][-1]:.3e}")
    with open(os.path.join(_EXPDIR, _HW_JSON), "w") as f:
        json.dump(result, f, indent=2)
    return result


# --------------------------------------------------------------------------
# Figure: simulation lines + 2^-n extrapolation + hardware sweep points
# --------------------------------------------------------------------------
def plot(sim=None, hw=None):
    if sim is None:
        sim, hw = load()
    from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter
    from matplotlib.lines import Line2D
    plt = plotting.set_house_style()
    C_GLOBAL, C_SUB = plotting.PALETTE["global"], plotting.PALETTE["method"]
    C_REF, C_FLOOR = plotting.PALETTE["reference"], plotting.PALETTE["floor"]

    ng = np.array(sim["n_global"], dtype=float)
    ns = np.array(sim["n_subspace"], dtype=float)
    nmax = max(N_LIST_HW)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    floor = (hw or {}).get("shot_floor_var", 1.0 / SHOTS)
    ax.axhspan(1e-9, floor, color=C_FLOOR, alpha=0.06, zorder=0)
    ax.axhline(floor, ls=(0, (1, 1.2)), lw=1.6, color=C_FLOOR, alpha=0.9, zorder=1)
    ax.text(nmax, floor * 1.35, "shot-noise floor", color=C_FLOOR, fontsize=9.5,
            ha="right", va="bottom", style="italic")

    n_ext = np.linspace(ng[0], nmax, 300)
    ref0 = sim["global_var"][0] * 2 ** ng[0]
    ax.plot(n_ext, ref0 * 2.0 ** (-n_ext), ls=(0, (5, 4)), lw=1.6, color=C_REF,
            alpha=0.9, zorder=2)

    ax.plot(ng, sim["global_var"], "-", lw=2.2, color=C_GLOBAL, zorder=3)
    ax.plot(ng, sim["global_var"], "o", ms=6.5, color=C_GLOBAL,
            markeredgecolor="white", markeredgewidth=0.7, zorder=4)
    ax.plot(ns, sim["subspace_var"], "-", lw=2.2, color=C_SUB, zorder=3)
    ax.plot(ns, sim["subspace_var"], "s", ms=6.0, color=C_SUB,
            markeredgecolor="white", markeredgewidth=0.7, zorder=4)

    handles = [
        Line2D([], [], color=C_GLOBAL, lw=2.2, marker="o", ms=6.5, mec="white",
               label=r"global ansatz (sim)"),
        Line2D([], [], color=C_SUB, lw=2.2, marker="s", ms=6.0, mec="white",
               label=rf"subspace ansatz (sim, $m={M}$)"),
        Line2D([], [], color=C_REF, lw=1.6, ls=(0, (5, 4)), label=r"$2^{-n}$ scaling"),
    ]

    if hw is not None:
        hn = np.array(hw["n"], dtype=float)
        ax.plot(hn, hw["global_var"], "*", ms=17, color=C_GLOBAL,
                markeredgecolor="k", markeredgewidth=0.7, zorder=6)
        ax.plot(hn, hw["subspace_var"], "*", ms=17, color=C_SUB,
                markeredgecolor="k", markeredgewidth=0.7, zorder=6)
        handles += [
            Line2D([], [], color=C_GLOBAL, lw=0, marker="*", ms=15, mec="k",
                   label=r"global on $\mathtt{ibm\_kobe}$"),
            Line2D([], [], color=C_SUB, lw=0, marker="*", ms=15, mec="k",
                   label=r"subspace on $\mathtt{ibm\_kobe}$"),
        ]

    ax.annotate(r"trainable: $\mathrm{Var}\sim\Omega(1/k)$",
                xy=(11.0, 0.046), xytext=(9.0, 0.30),
                fontsize=10, color=C_SUB, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=C_SUB, lw=1.1,
                                connectionstyle="arc3,rad=0.2"))
    ax.annotate(r"untrainable: $\mathrm{Var}\sim 2^{-n}$",
                xy=(14.0, sim["global_var"][6]), xytext=(42.0, 1.1e-5),
                fontsize=10, color=C_GLOBAL, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=C_GLOBAL, lw=1.1,
                                connectionstyle="arc3,rad=0.2"))

    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xlim(ng[0] * 0.85, nmax * 1.18)
    ax.set_ylim(min(floor * 0.25, 1e-7), 0.6)
    ax.set_xlabel(r"number of qubits $n$", fontsize=13)
    ax.set_ylabel(r"gradient variance $\mathrm{Var}\,[\partial_\theta \mathcal{L}]$", fontsize=13)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20))
    ax.grid(True, which="major", ls="-", lw=0.5, alpha=0.25)
    ax.grid(True, which="minor", ls=":", lw=0.4, alpha=0.15)
    ax.legend(handles=handles, fontsize=8.8, loc="lower left",
              bbox_to_anchor=(0.015, 0.015), frameon=True, framealpha=0.92,
              edgecolor="0.7", borderpad=0.6, labelspacing=0.45, handlelength=2.0)
    fig.tight_layout()
    plotting.savefig(fig, _FIG_STEM)
    plt.close(fig)


def load():
    with open(os.path.join(_EXPDIR, _SIM_JSON)) as f:
        sim = json.load(f)
    hw_path = os.path.join(_EXPDIR, _HW_JSON)
    hw = json.load(open(hw_path)) if os.path.exists(hw_path) else None
    return sim, hw
