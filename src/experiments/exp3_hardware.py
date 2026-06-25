"""
Experiment III — Hardware validation on IBM ibm_kobe.

Subspace circuits V = U_A S were compiled for k in {2,4,8,16} and executed
on ibm_kobe.  Hellinger fidelity is reported vs the ideal statevector output.
Results are hardcoded from the completed hardware run.

  from src.experiments.exp3_hardware import plot
"""
import numpy as np

from src.analysis import plotting

_FIG_STEM = "exp3_fidelity"

# ibm_kobe hardware results (backend: ibm_kobe, shots=4096)
DATA = {
    2:  {"qubits": 1, "cx": 0,   "fidelity": 1.0000000000000000},
    4:  {"qubits": 2, "cx": 2,   "fidelity": 0.9978418696449176},
    8:  {"qubits": 3, "cx": 29,  "fidelity": 0.9876384537527638},
    16: {"qubits": 4, "cx": 187, "fidelity": 0.8459914631359741},
}


def plot(save_pdf=False):
    plt = plotting.set_house_style()
    C_M = plotting.PALETTE["method"]
    fig, ax = plt.subplots(figsize=(5.5, 4.3))

    ks = sorted(DATA)
    fid  = [DATA[k]["fidelity"] for k in ks]
    twoq = [DATA[k]["cx"]       for k in ks]
    qb   = [DATA[k]["qubits"]   for k in ks]

    ax.plot(ks, fid, "o-", color=C_M, lw=2.0, ms=8, mec="white", mew=0.7)

    offsets = {2: ((10, -12), "center", "top"),
               4: ((0, -10),  "right",  "top"),
               8: ((-3, -7),  "right",  "top"),
               16: ((-30, 0), "center", "bottom")}
    for k, f, g, q in zip(ks, fid, twoq, qb):
        xy, ha, va = offsets[k]
        ax.annotate(f"{q}q, {g} CX", xy=(k, f), xytext=xy,
                    textcoords="offset points", ha=ha, va=va,
                    fontsize=8.5, color="0.3")

    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel(r"subspace rank $k$")
    ax.set_ylabel("Hellinger fidelity $F$")
    ax.grid(True, which="major", ls="-", lw=0.5, alpha=0.25)
    fig.tight_layout()
    plotting.savefig(fig, _FIG_STEM, pdf=save_pdf)
    plt.close(fig)
