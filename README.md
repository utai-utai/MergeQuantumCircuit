# MergeQuantumCircuit

Subspace quantization of near-unitary neural-network cores — a method that transfers
trained classical weights into quantum circuits analytically, with no quantum-side training.

## Method

Given a near-unitary residual weight W, the **transfer map** T(W) returns three objects:

| Symbol | Meaning |
|--------|---------|
| **Q** | Stiefel frame — top-k left singular vectors (k = 2^m, m qubits) |
| **U_A** | Nearest unitary to the projected subspace block |
| **H** | Hermitian generator — H = i log U_A |

The **mixing operator** O(Q, H) = Q e^{−iH} Q† approximates W with a tight error bound:

```
‖W − O‖_F  ≤  ‖W − Π_Q(W)‖_F  +  ‖P_A − I‖_F
               (truncation)        (non-unitarity)
```

The same construction supports **manifold merging**: two specialised cores W_A and W_B are
blended in the Lie-algebra generator space via a covering frame Q_C = orth([Q_A, Q_B]).

## Requirements

```bash
pip install torch qiskit qiskit-aer pennylane numpy matplotlib
```

IBM hardware experiments additionally require:

```bash
pip install qiskit-ibm-runtime
```

Set `IBM_API_KEY` and `IBM_CRN_STRING` (channel `ibm_cloud`) in a `.env` file at the project
root. The `.env` file is git-ignored and must never be committed.

## Quick start

```bash
jupyter notebook tutorial.ipynb
```

`tutorial.ipynb` at the project root walks through all eight experiments in sequence.
Exp III uses hardcoded IBM hardware data, Experiments IV and V load pre-computed
JSON results, and the rest run live (quick-demo parameters by default,
full-paper parameters noted as comments).

## Project structure

```
src/
  core/
    geometric_qml.py     central engine: transfer_map, error_decomposition, merge_generators
    data.py              MNIST loaders and near-identity weight initialiser
    hardware.py          IBM Quantum backend helpers (transpile, submit, fetch)
  experiments/
    validation.py        Val A–D: method verification on MNIST
    exp1_vit.py          Exp I:   ViT zero-shot residual core transfer
    exp2_dit.py          Exp II:  DiT quantum diffusion model (3 phases)
    exp3_hardware.py     Exp III: hardware fidelity on IBM ibm_kobe
    exp4_bp.py           Exp IV:  barren plateau suppression (n = 2 … 128 qubits)
    exp5_expr.py         Exp V:   effective dimension per two-qubit-gate budget
  analysis/
    metrics.py           Fisher information, effective dimension, Hellinger fidelity, TVD
    plotting.py          shared figure style and savefig helper
result/
  exp4_bp/               pre-computed JSON results for Exp IV
  exp5_expr/             pre-computed JSON results for Exp V
docs/
  THEORY.md              complete theory with proofs
  ARCHITECTURE.md        one-page architecture overview
tutorial.ipynb           interactive walkthrough (start here)
```

## Experiments

| | Name | Key result |
|---|------|------------|
| **Val A** | Zero-shot transfer | Quantum model matches classical accuracy; circuit cross-check exact |
| **Val B** | Error decomposition | Total ≤ truncation + non-unitarity (Theorem 4.2) |
| **Val C** | Task-informed initialization | (H_0=i\log U_A) vs identity and norm-matched random controls |
| **Val D** | Manifold merging | Generator-space blend outperforms weight averaging |
| **Exp I** | ViT residual core | Quantum zero-shot tracks classical accuracy epoch-by-epoch |
| **Exp II** | DiT diffusion | 3-phase: classical → zero-shot → generator fine-tune |
| **Exp III** | IBM hardware fidelity | Hellinger F stays high through k = 8; k = 16 drops to 0.846 at 187 CX |
| **Exp IV** | Barren plateaus | Subspace Var ~ Ω(1/k) vs global 2^{−n} (n = 2 … 128) |
| **Exp V** | Effective dimension | Method 42.1 vs PQC ceiling 25.4 (sim); 28.0 on ibm_kobe |
