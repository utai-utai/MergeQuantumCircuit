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
import csv
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler

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
# Validation 3 — task-informed generator initialisation
# ==========================================================================
def hermitian_basis(k, device=None):
    """A fixed orthonormal real basis of k x k Hermitian matrices.

    Every condition below optimises the same coefficients in this basis.  The
    only experimental variable is H_0, not the variational ansatz.
    """
    basis = []
    for i in range(k):
        E = torch.zeros((k, k), dtype=torch.complex64)
        E[i, i] = 1
        basis.append(E)
    scale = 1.0 / math.sqrt(2.0)
    for i in range(k):
        for j in range(i + 1, k):
            symmetric = torch.zeros((k, k), dtype=torch.complex64)
            symmetric[i, j] = symmetric[j, i] = scale
            basis.append(symmetric)
            antisymmetric = torch.zeros((k, k), dtype=torch.complex64)
            antisymmetric[i, j] = -1j * scale
            antisymmetric[j, i] = 1j * scale
            basis.append(antisymmetric)
    return torch.stack(basis).to(device=device)


def random_matched_hermitian(H_task, seed):
    """Random Hermitian with precisely the Frobenius norm of H_task."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    real = torch.randn(H_task.shape, generator=generator)
    imag = torch.randn(H_task.shape, generator=generator)
    A = (real + 1j * imag).to(torch.complex64)
    H = 0.5 * (A + A.mH)
    task_norm = torch.linalg.norm(H_task, ord="fro")
    return H * (task_norm / torch.linalg.norm(H, ord="fro"))


class TrainableGeneratorHybridNet(nn.Module):
    """Frozen classical surround with U(theta)=exp[-i(H_0 + sum theta_j V_j)]."""
    def __init__(self, classical_model, Q, H0, basis):
        super().__init__()
        self.flatten = copy.deepcopy(classical_model.flatten)
        self.fc_in = copy.deepcopy(classical_model.fc_in)
        self.fc_out = copy.deepcopy(classical_model.fc_out)
        self.relu = copy.deepcopy(classical_model.relu)
        for module in (self.fc_in, self.fc_out):
            for parameter in module.parameters():
                parameter.requires_grad = False
        self.register_buffer("Q", Q.to(torch.complex64))
        self.register_buffer("H0", H0.to(torch.complex64))
        self.register_buffer("basis", basis.to(torch.complex64))
        self.theta = nn.Parameter(torch.zeros(basis.shape[0]))

    def forward(self, x):
        x = self.relu(self.fc_in(self.flatten(x)))
        H = self.H0 + torch.einsum("j,jab->ab", self.theta.to(self.basis.dtype), self.basis)
        U = torch.linalg.matrix_exp(-1j * H)
        O_T = (self.Q @ U @ self.Q.mH).T
        x_q = (x.to(torch.complex64) @ O_T).real
        return self.fc_out(self.relu(x + x_q))


def _seeded_batches(dataset, batch_size, seed, steps):
    """Identical per-seed batch order for all three initialisation conditions."""
    generator = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(dataset, replacement=True,
                            num_samples=steps * batch_size, generator=generator)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      num_workers=0, drop_last=True)


def _initial_metrics(model, batch, criterion):
    data_, target = (item.to(DEVICE) for item in batch)
    model.train()
    model.zero_grad(set_to_none=True)
    loss = criterion(model(data_), target)
    loss.backward()
    grad_norm = model.theta.grad.norm().item()
    return loss.item(), grad_norm


def _mean_std(records, field):
    values = torch.tensor([record[field] for record in records], dtype=torch.float64)
    return values.mean().item(), values.std(unbiased=False).item()


def experiment_task_informed_initialization(
        train_loader, test_loader, hidden_dim=64, k=16, seeds=(0, 1, 2, 3, 4),
        steps=200, eval_interval=5, lr=1e-2, fine_tune_batch_size=None,
        output_name="val3_initialization"):
    """Validation III: isolate the effect of task-informed H_0.

    Q, k, the frozen classical surround, the Hermitian basis {V_j}, optimiser,
    and each seed's training batches are shared by task-informed, identity, and
    norm-matched random H_0.  ``fine_tune_batch_size`` changes only the quantum
    fine-tuning batches, not the classical pre-training. Artefacts are written
    to result/val3_initialization/.
    """
    if k != 16:
        raise ValueError("Validation III is specified for the fixed active-space rank k=16.")
    if steps <= 0 or eval_interval <= 0:
        raise ValueError("steps and eval_interval must be positive.")
    fine_tune_batch_size = fine_tune_batch_size or train_loader.batch_size

    print("\n" + "=" * 60 +
          f"\nValidation III: task-informed initialisation (k={k}, steps={steps})")
    # This is deliberately trained once: all conditions share exactly one Q and H_task.
    torch.manual_seed(SEED)
    base = train_model(ClassicalResNet(hidden_dim).to(DEVICE), train_loader, epochs=3)
    W = base.core_layer.weight.detach().cpu()
    Q, _, H_task = transfer_map(W, k)
    basis = hermitian_basis(k)
    conditions = ("task_informed", "identity", "random_matched")
    output_dir = os.path.join(RESULT_DIR, output_name)
    os.makedirs(output_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    all_records, curves = [], []

    for seed in seeds:
        batches = list(_seeded_batches(train_loader.dataset, fine_tune_batch_size,
                                       seed, steps))
        for condition in conditions:
            if condition == "task_informed":
                H0 = H_task
            elif condition == "identity":
                H0 = torch.zeros_like(H_task)
            else:
                H0 = random_matched_hermitian(H_task, seed)
            model = TrainableGeneratorHybridNet(base, Q, H0, basis).to(DEVICE)
            optimizer = optim.Adam([model.theta], lr=lr)
            initial_loss, gradient_norm = _initial_metrics(model, batches[0], criterion)
            initial_accuracy = evaluate(model, test_loader)
            record = {"condition": condition, "seed": seed,
                      "initial_accuracy": initial_accuracy, "initial_loss": initial_loss,
                      "initial_gradient_norm": gradient_norm,
                      "h0_frobenius_norm": torch.linalg.norm(H0, ord="fro").item()}
            all_records.append(record)
            curves.append({"condition": condition, "seed": seed, "step": 0,
                           "train_loss": initial_loss, "test_accuracy": initial_accuracy})

            for step, batch in enumerate(batches, start=1):
                data_, target = (item.to(DEVICE) for item in batch)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(data_), target)
                loss.backward()
                optimizer.step()
                if step % eval_interval == 0 or step == steps:
                    curves.append({"condition": condition, "seed": seed, "step": step,
                                   "train_loss": loss.item(),
                                   "test_accuracy": evaluate(model, test_loader)})
            print(f"  seed={seed}  {condition:15s}  step-0 acc={initial_accuracy:6.2f}%  "
                  f"loss={initial_loss:.4f}  |grad|={gradient_norm:.4e}")

    summary = {}
    for condition in conditions:
        records = [record for record in all_records if record["condition"] == condition]
        summary[condition] = {
            field: dict(zip(("mean", "std"), _mean_std(records, field)))
            for field in ("initial_accuracy", "initial_loss", "initial_gradient_norm")
        }
    with open(os.path.join(output_dir, "initial_metrics.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_records[0]))
        writer.writeheader(); writer.writerows(all_records)
    with open(os.path.join(output_dir, "curves.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0]))
        writer.writeheader(); writer.writerows(curves)
    with open(os.path.join(output_dir, "summary.json"), "w") as handle:
        json.dump({"configuration": {"k": k, "steps": steps, "eval_interval": eval_interval,
                                      "lr": lr, "seeds": list(seeds), "num_generators": k * k,
                                      "fine_tune_batch_size": fine_tune_batch_size},
                   "summary": summary}, handle, indent=2)
    _plot_initialization_curves(curves, output_dir)
    print("\nStep-0 mean +/- std over seeds:")
    for condition, values in summary.items():
        print(f"  {condition:15s} acc={values['initial_accuracy']['mean']:.2f} +/- "
              f"{values['initial_accuracy']['std']:.2f}%  loss={values['initial_loss']['mean']:.4f} +/- "
              f"{values['initial_loss']['std']:.4f}  |grad|={values['initial_gradient_norm']['mean']:.3e} +/- "
              f"{values['initial_gradient_norm']['std']:.3e}")
    return summary


def _plot_initialization_curves(curves, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colours = {"task_informed": "#1f3b73", "identity": "#2a8c4a", "random_matched": "#b22222"}
    labels = {"task_informed": "Task-informed $H_0=i\\log U_A$",
              "identity": "Identity $H_0=0$", "random_matched": "Random matched-norm $H_0$"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for condition in colours:
        rows = [row for row in curves if row["condition"] == condition]
        steps = sorted({row["step"] for row in rows})
        for axis, field, ylabel in zip(axes, ("train_loss", "test_accuracy"),
                                       ("Train loss", "Test accuracy (%)")):
            means, stds = [], []
            for step in steps:
                values = [row[field] for row in rows if row["step"] == step]
                means.append(sum(values) / len(values))
                stds.append(torch.tensor(values).std(unbiased=False).item())
            axis.plot(steps, means, lw=2.2, color=colours[condition], label=labels[condition])
            axis.fill_between(steps, [a - b for a, b in zip(means, stds)],
                              [a + b for a, b in zip(means, stds)], color=colours[condition], alpha=.16)
            axis.set_xlabel("Optimisation steps")
            axis.set_ylabel(ylabel)
            axis.grid(True, ls="--", alpha=.4)
    axes[1].legend(fontsize=9, framealpha=.95)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "learning_curves.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "learning_curves.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Paper-ready early-optimisation view: the regime where initialisation can matter.
    fig, axis = plt.subplots(figsize=(6.4, 4.5))
    for condition in colours:
        rows = [row for row in curves if row["condition"] == condition and 1 <= row["step"] <= 50]
        steps = sorted({row["step"] for row in rows})
        means, stds = [], []
        for step in steps:
            values = [row["test_accuracy"] for row in rows if row["step"] == step]
            means.append(sum(values) / len(values))
            stds.append(torch.tensor(values).std(unbiased=False).item())
        axis.plot(steps, means, lw=2.5, marker="o", ms=3.5,
                  color=colours[condition], label=labels[condition])
        axis.fill_between(steps, [a - b for a, b in zip(means, stds)],
                          [a + b for a, b in zip(means, stds)],
                          color=colours[condition], alpha=.16)
    axis.set(xlim=(5, 50), xlabel="Optimisation steps", ylabel="Test accuracy (%)")
    axis.grid(True, ls="--", alpha=.4)
    axis.legend(fontsize=8.5, framealpha=.95, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "early_accuracy_steps_1_50.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "early_accuracy_steps_1_50.pdf"), bbox_inches="tight")
    plt.close(fig)


def hardware_local_generator_family(family="full15", num_qubits=4):
    """Fixed, symmetric hardware-native control families on a 4-qubit register."""
    I = torch.eye(2, dtype=torch.complex64)
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

    def pauli_string(operators):
        out = operators[0]
        for operator in operators[1:]:
            out = torch.kron(out, operator)
        return out / math.sqrt(2 ** num_qubits)  # unit Frobenius norm

    controls = []
    if family not in {"ry4", "ryrz8", "full15"}:
        raise ValueError(f"Unknown generator family: {family}")
    for qubit in range(num_qubits):
        paulis = (Y,) if family == "ry4" else (Y, Z) if family == "ryrz8" else (X, Y, Z)
        for pauli in paulis:
            operators = [I] * num_qubits
            operators[qubit] = pauli
            controls.append(pauli_string(operators))
    if family == "full15":
        for qubit in range(num_qubits - 1):
            operators = [I] * num_qubits
            operators[qubit] = Z
            operators[qubit + 1] = Z
            controls.append(pauli_string(operators))
    return torch.stack(controls)


def _paired_statistics(values, bootstrap_samples=10000, seed=2026):
    """Mean, percentile bootstrap CI, and exact two-sided sign-permutation p-value."""
    delta = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(delta), (bootstrap_samples, len(delta)), generator=generator)
    boot_means = delta[indices].mean(dim=1)
    ci = torch.quantile(boot_means, torch.tensor([0.025, 0.975], dtype=torch.float64))
    observed = abs(delta.mean()).item()
    signs = torch.tensor([[1 if (mask >> bit) & 1 else -1 for bit in range(len(delta))]
                          for mask in range(2 ** len(delta))], dtype=torch.float64)
    null_means = (signs * delta).mean(dim=1).abs()
    return {"mean": delta.mean().item(), "std": delta.std(unbiased=False).item(),
            "ci95_low": ci[0].item(), "ci95_high": ci[1].item(),
            "permutation_p_two_sided": (null_means >= observed - 1e-12).double().mean().item()}


def _plot_independent_initialisation(curves, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colours = {"task_informed": "#1f3b73", "identity": "#2a8c4a", "random_matched": "#b22222"}
    labels = {"task_informed": "Task-informed $H_0=i\\log U_A$",
              "identity": "Identity $H_0=0$", "random_matched": "Random matched-norm $H_0$"}
    fig, axis = plt.subplots(figsize=(7.1, 4.7))
    for condition in colours:
        rows = [row for row in curves if row["condition"] == condition and row["step"] <= 50]
        steps = sorted({row["step"] for row in rows})
        means, stds = [], []
        for step in steps:
            values = [row["test_accuracy"] for row in rows if row["step"] == step]
            means.append(sum(values) / len(values))
            stds.append(torch.tensor(values).std(unbiased=False).item())
        axis.plot(steps, means, lw=2.3, color=colours[condition], label=labels[condition])
        axis.fill_between(steps, [a - b for a, b in zip(means, stds)],
                          [a + b for a, b in zip(means, stds)], color=colours[condition], alpha=.16)
    axis.set(xlim=(0, 50), xlabel="Optimisation steps", ylabel="Test accuracy (%)")
    axis.grid(True, ls="--", alpha=.4)
    axis.legend(fontsize=8.8, framealpha=.95, loc="lower right")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"accuracy_steps_0_50.{extension}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


def experiment_task_informed_initialization_independent(
        train_loader, test_loader, hidden_dim=64, k=16, seeds=(0, 1, 2, 3, 4),
        steps=50, fine_tune_batch_size=32, lr=1e-2, accuracy_threshold=94.4,
        output_name="val3_ini", generator_family="full15"):
    """Preregistered paired Validation III with independently trained base models.

    Each seed trains a separate classical base and independently extracts
    (W_s, Q_s, H_task,s).  The three H_0 conditions are then paired within the
    seed through the same fixed control family and precisely the same batches.
    """
    if k != 16:
        raise ValueError("Validation III fixes the active-space rank at k=16.")
    if steps != 50:
        raise ValueError("This preregistered validation records every step from 0 through 50.")
    basis = hardware_local_generator_family(generator_family, num_qubits=4)
    conditions = ("task_informed", "identity", "random_matched")
    output_dir = os.path.join(RESULT_DIR, output_name)
    os.makedirs(output_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    initial_records, curves, scalar_records = [], [], []
    print("\n" + "=" * 60 +
          f"\nValidation III: independent bases + {basis.shape[0]}-control {generator_family} family")

    for seed in seeds:
        # Both the classical weights and its shuffled training order are independently seeded.
        torch.manual_seed(seed)
        base = train_model(ClassicalResNet(hidden_dim).to(DEVICE), train_loader, epochs=3)
        W = base.core_layer.weight.detach().cpu()
        Q, _, H_task = transfer_map(W, k)
        batches = list(_seeded_batches(train_loader.dataset, fine_tune_batch_size,
                                       10000 + seed, steps))
        per_condition_curves = {}
        for condition in conditions:
            H0 = (H_task if condition == "task_informed" else torch.zeros_like(H_task)
                  if condition == "identity" else random_matched_hermitian(H_task, 20000 + seed))
            model = TrainableGeneratorHybridNet(base, Q, H0, basis).to(DEVICE)
            optimizer = optim.Adam([model.theta], lr=lr)
            initial_loss, gradient_norm = _initial_metrics(model, batches[0], criterion)
            initial_accuracy = evaluate(model, test_loader)
            initial_records.append({"condition": condition, "seed": seed,
                                    "initial_accuracy": initial_accuracy, "initial_loss": initial_loss,
                                    "initial_gradient_norm": gradient_norm,
                                    "h_task_frobenius_norm": torch.linalg.norm(H_task, ord="fro").item(),
                                    "h0_frobenius_norm": torch.linalg.norm(H0, ord="fro").item()})
            run_curve = [{"condition": condition, "seed": seed, "step": 0,
                          "train_loss": initial_loss, "test_accuracy": initial_accuracy}]
            for step, batch in enumerate(batches, start=1):
                data_, target = (item.to(DEVICE) for item in batch)
                model.train(); optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(data_), target)
                loss.backward(); optimizer.step()
                run_curve.append({"condition": condition, "seed": seed, "step": step,
                                  "train_loss": loss.item(), "test_accuracy": evaluate(model, test_loader)})
            curves.extend(run_curve)
            per_condition_curves[condition] = run_curve
            print(f"  seed={seed} {condition:15s} step-0 acc={initial_accuracy:6.2f}% "
                  f"loss={initial_loss:.4f} |grad|={gradient_norm:.3e}")

        for condition in conditions:
            run_curve = per_condition_curves[condition]
            steps_t = torch.tensor([row["step"] for row in run_curve], dtype=torch.float64)
            accuracies = torch.tensor([row["test_accuracy"] for row in run_curve], dtype=torch.float64)
            hits = [row["step"] for row in run_curve if row["test_accuracy"] >= accuracy_threshold]
            scalar_records.append({"condition": condition, "seed": seed,
                                   "accuracy_auc_0_50": torch.trapz(accuracies, steps_t).item(),
                                   "steps_to_threshold": hits[0] if hits else steps + 1,
                                   "reached_threshold": bool(hits)})

    paired = {}
    for baseline in ("identity", "random_matched"):
        task = {row["seed"]: row for row in scalar_records if row["condition"] == "task_informed"}
        control = {row["seed"]: row for row in scalar_records if row["condition"] == baseline}
        auc_delta = [task[seed]["accuracy_auc_0_50"] - control[seed]["accuracy_auc_0_50"] for seed in seeds]
        threshold_delta = [task[seed]["steps_to_threshold"] - control[seed]["steps_to_threshold"] for seed in seeds]
        paired[baseline] = {"auc_task_minus_baseline": _paired_statistics(auc_delta, seed=30000),
                            "steps_to_threshold_task_minus_baseline": _paired_statistics(threshold_delta, seed=40000)}
    config = {"k": k, "seeds": list(seeds), "steps": steps, "fine_tune_batch_size": fine_tune_batch_size,
              "lr": lr, "accuracy_threshold": accuracy_threshold, "num_generators": basis.shape[0],
              "generator_family": generator_family,
              "test_evaluation_steps": "every step, including step 0"}
    for name, rows in (("initial_metrics.csv", initial_records), ("curves.csv", curves),
                       ("scalar_metrics.csv", scalar_records)):
        with open(os.path.join(output_dir, name), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    with open(os.path.join(output_dir, "paired_statistics.json"), "w") as handle:
        json.dump({"configuration": config, "paired_results": paired}, handle, indent=2)
    _plot_independent_initialisation(curves, output_dir)
    return paired


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
def run(hidden_dim=64, k=16, save_pdf=False, include_task_initialization=False):
    from src.core.data import mnist_split_loaders
    train_all, train_0_4, train_5_9, test = mnist_split_loaders()
    W = experiment_A(train_all, test, hidden_dim=hidden_dim, k=k)
    experiment_B(W, save_pdf=save_pdf)
    experiment_C(train_all, train_0_4, train_5_9, test, hidden_dim=hidden_dim, k=k)
    if include_task_initialization:
        experiment_task_informed_initialization(train_all, test, hidden_dim=hidden_dim, k=k)
