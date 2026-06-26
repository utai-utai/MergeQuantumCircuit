"""
Method showcase (docs/THEORY.md) — the corrected classical->quantum transfer map
and manifold merging on MNIST.  Three demonstrations, all writing to result/:

  Experiment A — zero-shot quantum transfer: train a classical residual core,
      replace it by O(Q,H)=Q e^{-iH} Q^H (no quantum-side training), compare
      accuracy, and cross-check the matrix operator against a PennyLane circuit.
  Experiment B — error decomposition vs rank k (the central Method figure):
      ||W-O||_F <= ||W-Pi_Q(W)||_F (truncation) + ||P_A-I||_F (non-unitarity).
  Experiment C — manifold merging of two specialised cores in the Lie algebra
      over the covering frame Q_C = orth([Q_A,Q_B]).

  from src.experiments.val1_transfer import run   # A + B + C
"""
import os
import math
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim

from src import RESULT_DIR, SEED
from src.core.data import mnist_split_loaders, near_identity_weight
from src.core.geometric_qml import (
    transfer_map, mixing_operator, error_decomposition, merge_generators,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================================
# Models: residual near-identity core + its quantum (mixing-operator) version
# ==========================================================================
class ClassicalResNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc_in = nn.Linear(28 * 28, hidden_dim)
        self.core_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)
        with torch.no_grad():
            self.core_layer.weight.copy_(near_identity_weight(hidden_dim))
        self.fc_out = nn.Linear(hidden_dim, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc_in(self.flatten(x)))
        x = x + self.core_layer(x)          # residual near-identity block (I + W)
        return self.fc_out(self.relu(x))


class QuantumHybridNet(nn.Module):
    """Classical model with the residual core replaced by O = Q U Q^H
    (U = e^{-iH}).  Applying O as a matrix is identical to the
    StatePrep + QubitUnitary circuit (verified in crosscheck_against_circuit)."""

    def __init__(self, classical_model, Q, U):
        super().__init__()
        self.flatten = classical_model.flatten
        self.fc_in = classical_model.fc_in
        self.fc_out = classical_model.fc_out
        self.relu = classical_model.relu
        O = mixing_operator(Q, U)
        self.register_buffer("O_T", O.T.contiguous())
        self.k = Q.shape[1]

    def forward(self, x):
        x = self.relu(self.fc_in(self.flatten(x)))
        x_q = (x.to(torch.complex64) @ self.O_T).real
        return self.fc_out(self.relu(x + x_q))


# ==========================================================================
# Train / eval
# ==========================================================================
def train_model(model, loader, epochs=3, freeze_surrounding=False, identity_reg=1e-3):
    model.to(DEVICE)
    if freeze_surrounding:
        for name, p in model.named_parameters():
            if "core_layer" not in name:
                p.requires_grad = False
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for data_, target in loader:
            data_, target = data_.to(DEVICE), target.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(data_), target)
            if identity_reg:                          # soft anchor: keep W near-unitary
                W = model.core_layer.weight
                loss = loss + identity_reg * nn.functional.mse_loss(
                    W, torch.eye(W.shape[0], device=W.device))
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for data_, target in loader:
        data_, target = data_.to(DEVICE), target.to(DEVICE)
        correct += (model(data_).argmax(dim=1) == target).sum().item()
        total += target.size(0)
    return 100.0 * correct / total


# ==========================================================================
# PennyLane cross-check: the matrix operator == the physical circuit
# ==========================================================================
def crosscheck_against_circuit(Q, U_A, k):
    import pennylane as qml
    num_qubits = int(math.log2(k))
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(state, U):
        qml.StatePrep(state, wires=range(num_qubits))
        qml.QubitUnitary(U, wires=range(num_qubits))
        return qml.state()

    torch.manual_seed(0)
    x_sub = nn.functional.normalize(torch.rand(k, dtype=torch.complex64), dim=0)
    circ_out = circuit(x_sub, U_A.cpu())
    mat_out = x_sub @ U_A.cpu().T
    max_dev = torch.max(torch.abs(circ_out - mat_out)).item()
    diagram = qml.draw(circuit)(x_sub, U_A.cpu())
    print(f"Compiled subspace unitary on {num_qubits} qubits (k={k}):\n")
    print(diagram)
    print(f"\nmax |circuit - matrix| = {max_dev:.2e}  "
          f"(matrix operator O = Q U_A Q^H reproduces the physical circuit)")
    return max_dev, diagram


# ==========================================================================
# Validation 1 — zero-shot quantum transfer
# ==========================================================================
def experiment_A(train_loader, test_loader, hidden_dim=64, k=16):
    print("\n" + "=" * 60 + f"\nExperiment A: Zero-shot quantum transfer (k={k})")
    model = train_model(ClassicalResNet(hidden_dim).to(DEVICE), train_loader, epochs=3)
    acc_c = evaluate(model, test_loader)

    W = model.core_layer.weight.data.clone().cpu()
    Q, U_A, H = transfer_map(W, k)
    max_dev, _ = crosscheck_against_circuit(Q, U_A, k)
    print(f"[Circuit] compiled on {int(math.log2(k))} qubits; max|circuit-matrix| = {max_dev:.2e}")

    q_model = QuantumHybridNet(model, Q.to(DEVICE), U_A.to(DEVICE)).to(DEVICE)
    acc_q = evaluate(q_model, test_loader)
    tot, tr, nu = error_decomposition(W, k)
    lines = [
        "=== Experiment A: Zero-shot quantum transfer ===",
        f"hidden_dim={hidden_dim}  k={k}  qubits={int(math.log2(k))}",
        f"[Classical baseline]        Accuracy: {acc_c:.2f}%",
        f"[Quantum zero-shot O(Q,H)]  Accuracy: {acc_q:.2f}%",
        f"transfer error ||W-O||_F={tot:.4f}  (truncation={tr:.4f}, non-unitarity={nu:.4f})",
        f"circuit/matrix agreement: max deviation = {max_dev:.2e}",
    ]
    print("\n".join(lines[2:]))
    return W


# ==========================================================================
# Validation 2 — error decomposition vs rank k (central Method figure)
# ==========================================================================
def experiment_B(W, k_list=(2, 4, 8, 16, 32, 64), save_pdf=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    print("\n" + "=" * 60 + "\nExperiment B: Error decomposition vs rank k")
    k_list = [k for k in k_list if k <= W.shape[0]]
    totals, truncs, nonunits = [], [], []
    lines = ["=== Experiment B: Error decomposition vs rank k ===",
             f"{'k':>4} {'total':>10} {'truncation':>12} {'non-unitarity':>14} {'bound(tr+nu)':>14}"]
    for k in k_list:
        tot, tr, nu = error_decomposition(W, k)
        totals.append(tot); truncs.append(tr); nonunits.append(nu)
        lines.append(f"{k:>4} {tot:>10.4f} {tr:>12.4f} {nu:>14.4f} {tr + nu:>14.4f}")
        print("  " + lines[-1])

    plt.figure(figsize=(8, 5.2))
    plt.plot(k_list, totals,   "o-", lw=2.4, ms=7, color="#1f3b73", label=r"total  $\|W-\mathcal{O}\|_F$")
    plt.plot(k_list, truncs,   "s--", lw=2.0, ms=6, color="#2a8c4a", label=r"truncation  $\|W-\Pi_Q(W)\|_F$")
    plt.plot(k_list, nonunits, "^--", lw=2.0, ms=6, color="#b22222", label=r"non-unitarity  $\|P_A-I\|_F$")
    plt.xscale("log", base=2)
    plt.xticks(k_list, [str(k) for k in k_list])
    plt.xlabel(r"Truncation rank $k$  ($\log_2 k$ qubits)", fontsize=13)
    plt.ylabel("Frobenius error", fontsize=13)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11, framealpha=0.95)
    # plt.annotate("full rank: truncation $\\to$ 0,\ntotal $=$ non-unitarity floor",
    #              xy=(k_list[-1], totals[-1]),
    #              xytext=(k_list[len(k_list) // 2], max(totals) * 0.62),
    #              arrowprops=dict(arrowstyle="->", lw=1.3), fontsize=10, ha="center")
    plt.tight_layout()
    fig_path = os.path.join(RESULT_DIR, "val_error_decomposition.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    if save_pdf:
        plt.savefig(os.path.join(RESULT_DIR, "val_error_decomposition.pdf"), bbox_inches="tight")
    plt.close()


# ==========================================================================
# Validation 3 — manifold merging
# ==========================================================================
def experiment_C(train_all, train_0_4, train_5_9, test_loader, hidden_dim=64, k=16):
    print("\n" + "=" * 60 + f"\nExperiment C: Manifold merging (k={k})")
    base = train_model(ClassicalResNet(hidden_dim).to(DEVICE), train_all, epochs=3)
    model_A = train_model(copy.deepcopy(base), train_0_4, epochs=1, freeze_surrounding=True)
    model_B = train_model(copy.deepcopy(base), train_5_9, epochs=1, freeze_surrounding=True)
    W_A = model_A.core_layer.weight.data.clone().cpu()
    W_B = model_B.core_layer.weight.data.clone().cpu()

    model_C = copy.deepcopy(base)
    model_C.core_layer.weight.data = (0.5 * W_A + 0.5 * W_B).to(DEVICE)
    acc_classic = evaluate(model_C, test_loader)

    Q_C, H_C, U_C = merge_generators(W_A, W_B, k)
    q_merged = QuantumHybridNet(base, Q_C.to(DEVICE), U_C.to(DEVICE)).to(DEVICE)
    acc_quantum = evaluate(q_merged, test_loader)

    lines = [
        "=== Experiment C: Manifold merging ===",
        f"k={k}  covering frame dim k_C={Q_C.shape[1]} (<= 2k)",
        f"[Classical direct averaging] Accuracy: {acc_classic:.2f}%",
        f"[Quantum manifold merge]     Accuracy: {acc_quantum:.2f}%",
        "generator separation ||H_A'-H_B'||_F drives the O(.^2) merge penalty.",
    ]
    print("\n".join(lines[2:]))


# ==========================================================================
# Entry point
# ==========================================================================
def run(hidden_dim=64, k=16, save_pdf=False):
    from src.core.data import mnist_split_loaders
    train_all, train_0_4, train_5_9, test = mnist_split_loaders()
    W = experiment_A(train_all, test, hidden_dim=hidden_dim, k=k)
    experiment_B(W, save_pdf=save_pdf)
    experiment_C(train_all, train_0_4, train_5_9, test, hidden_dim=hidden_dim, k=k)
