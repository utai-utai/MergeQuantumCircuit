"""Quantitative Validation A--C for the conditional Mini-DiT transfer experiment.

This module intentionally contains only the requested coupling, conditional
accuracy, and generator-only recovery checks.  It uses three independently
trained classical checkpoints and pairs every quantum condition within seed.
"""
import csv
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import RESULT_DIR
from src.core.data import mnist_loaders
from src.experiments.exp2_dit import ConditionalMiniDiT, CosineDDPM


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FrozenMNISTClassifier(nn.Module):
    """Independent classifier used only to score conditional diffusion samples."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.head(self.features(x))


def _write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    return tensor.mean().item(), tensor.std(unbiased=False).item()


def _model():
    return ConditionalMiniDiT(img_size=32, patch_size=4, dim=192, depth=6, heads=4)


def train_classifier(train_loader, device, output_dir, epochs=5):
    path = os.path.join(output_dir, "frozen_mnist_classifier.pt")
    model = FrozenMNISTClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            F.cross_entropy(model(x), y).backward()
            optimizer.step()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    torch.save(model.state_dict(), path)
    return model


@torch.no_grad()
def classifier_test_accuracy(model, test_loader, device):
    """Held-out accuracy of the independently trained scoring classifier."""
    model.eval()
    correct = total = 0
    for x, y in test_loader:
        prediction = model(x.to(device)).argmax(dim=1)
        correct += (prediction.cpu() == y).sum().item()
        total += len(y)
    return correct / total


def make_fixed_test_cases(test_loader, timesteps, repeats=4, seed=314159):
    """Pre-generate four (t, epsilon) pairs per full-test image on CPU."""
    images, labels = [], []
    for x, y in test_loader:
        images.append(x.cpu())
        labels.append(y.cpu())
    images, labels = torch.cat(images), torch.cat(labels)
    generator = torch.Generator().manual_seed(seed)
    shape = (repeats,) + tuple(images.shape)
    return {
        "images": images,
        "labels": labels,
        "t": torch.randint(0, timesteps, (repeats, len(images)), generator=generator),
        "noise": torch.randn(shape, generator=generator),
        "repeats": repeats,
    }


@torch.no_grad()
def fixed_denoising_mse(model, diffusion, fixed_cases, device, batch_size=256):
    model.eval()
    total_squared, total_count = 0.0, 0
    images, labels = fixed_cases["images"], fixed_cases["labels"]
    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        x0, y = images[start:stop].to(device), labels[start:stop].to(device)
        for repeat in range(fixed_cases["repeats"]):
            t = fixed_cases["t"][repeat, start:stop].to(device)
            noise = fixed_cases["noise"][repeat, start:stop].to(device)
            prediction = model(diffusion.q_sample(x0, t, noise), t, y)
            total_squared += (prediction - noise).square().sum().item()
            total_count += prediction.numel()
    return total_squared / total_count


def train_classical_checkpoint(seed, loader, diffusion, device, epochs, output_dir):
    torch.manual_seed(seed)
    model = _model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=device)
            noise = torch.randn_like(x)
            loss_mse = F.mse_loss(model(diffusion.q_sample(x, t, noise), t, y), noise)
            reg = sum(block.core_layer.get_reg_loss() for block in model.blocks)
            optimizer.zero_grad()
            (loss_mse + 1e-3 * reg).backward()
            optimizer.step()
        print(f"  classical seed={seed} epoch={epoch}/{epochs} train-MSE={loss_mse.item():.4f}")
    path = os.path.join(output_dir, f"classical_seed_{seed}.pt")
    torch.save(model.state_dict(), path)
    return path


def load_quantum_from_checkpoint(path, device, gamma, k=16):
    model = _model().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.convert_to_quantum(k=k, gamma=gamma)
    return model.to(device)


def _freeze_except_generators(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "H_param" in name


def generator_finetune(model, loader, diffusion, fixed_cases, device, epochs, learning_rate=5e-5):
    _freeze_except_generators(model)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    curve = [{"epoch": 0, "mse": fixed_denoising_mse(model, diffusion, fixed_cases, device)}]
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=device)
            noise = torch.randn_like(x)
            optimizer.zero_grad()
            loss = F.mse_loss(model(diffusion.q_sample(x, t, noise), t, y), noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
            optimizer.step()
        curve.append({"epoch": epoch, "mse": fixed_denoising_mse(model, diffusion, fixed_cases, device)})
        print(f"  quantum fine-tune epoch={epoch}/{epochs} held-out-MSE={curve[-1]['mse']:.6f}")
    return curve


@torch.no_grad()
def conditional_metrics(model, classifier, diffusion, device, noise_seed,
                        samples_per_digit=50, batch_size=100):
    """Conditional accuracy and prediction coverage for one generated sample set."""
    model.eval(); classifier.eval()
    torch.manual_seed(noise_seed)
    labels = torch.arange(10, device=device).repeat_interleave(samples_per_digit)
    correct = total = 0
    correct_by_digit = torch.zeros(10, dtype=torch.long)
    count_by_digit = torch.zeros(10, dtype=torch.long)
    predicted_count = torch.zeros(10, dtype=torch.long)
    for start in range(0, len(labels), batch_size):
        y = labels[start:start + batch_size]
        x = torch.randn(len(y), 1, 32, 32, device=device)
        for t_value in reversed(range(diffusion.timesteps)):
            t = torch.full((len(y),), t_value, device=device, dtype=torch.long)
            x = diffusion.p_sample(model, x, t, y)
        # Both the diffusion model and classifier use MNIST's [-1, 1] normalisation.
        prediction = classifier(torch.clamp(x, -1, 1)).argmax(dim=1)
        correct += (prediction == y).sum().item()
        total += len(y)
        for digit in range(10):
            mask = y == digit
            count_by_digit[digit] += mask.sum().cpu()
            correct_by_digit[digit] += ((prediction == y) & mask).sum().cpu()
            predicted_count[digit] += (prediction == digit).sum().cpu()
    per_digit = (correct_by_digit.float() / count_by_digit.clamp_min(1)).tolist()
    return {
        "conditional_accuracy": correct / total,
        "predicted_class_coverage": (predicted_count > 0).float().mean().item(),
        "per_digit_accuracy": json.dumps(per_digit),
    }


def _plot_results(zero_rows, conditional_rows, finetune_rows, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gammas = [0.0, 0.1, 0.3, 1.0]
    classical = [row["classical_mse"] for row in zero_rows]
    means, stds = [], []
    for gamma in gammas:
        values = [row["mse"] for row in zero_rows if row["gamma"] == gamma]
        mean, std = _mean_std(values); means.append(mean); stds.append(std)
    fig, axis = plt.subplots(figsize=(6.6, 4.4))
    axis.errorbar(gammas, means, yerr=stds, fmt="o-", lw=2.2, capsize=4, color="#1f3b73", label="Zero-shot quantum")
    for seed in sorted({row["seed"] for row in zero_rows}):
        axis.plot(gammas, [next(row["mse"] for row in zero_rows if row["seed"] == seed and row["gamma"] == gamma) for gamma in gammas],
                  color="#1f3b73", alpha=.22, lw=1)
    axis.axhline(sum(classical) / len(classical), color="#333333", ls="--", label="Classical")
    axis.set(xlabel=r"Residual coupling $\gamma$", ylabel="Fixed-noise test denoising MSE")
    axis.grid(True, ls="--", alpha=.4); axis.legend(); fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(os.path.join(output_dir, f"fig_d1_coupling_mse.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    phases = ["classical", "zero_gamma_1", "zero_gamma_0.1", "fine_tuned"]
    labels = ["Classical", r"Zero-shot $\gamma=1.0$", r"Zero-shot $\gamma=0.1$", "Fine-tuned quantum"]
    phase_means, phase_stds = [], []
    for phase in phases:
        mean, std = _mean_std([row["conditional_accuracy"] for row in conditional_rows if row["phase"] == phase])
        phase_means.append(mean); phase_stds.append(std)
    fig, axis = plt.subplots(figsize=(7.4, 4.4))
    axis.bar(range(4), phase_means, yerr=phase_stds, capsize=4, color=["#555555", "#1f3b73", "#528f45", "#8c3c86"])
    axis.set(xticks=range(4), xticklabels=labels, ylabel="Independent-classifier conditional accuracy", ylim=(0, 1))
    axis.tick_params(axis="x", labelrotation=14); axis.grid(True, axis="y", ls="--", alpha=.4); fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(os.path.join(output_dir, f"fig_d2_conditional_accuracy.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    epochs = sorted({row["epoch"] for row in finetune_rows})
    means, stds = [], []
    for epoch in epochs:
        mean, std = _mean_std([row["mse"] for row in finetune_rows if row["epoch"] == epoch])
        means.append(mean); stds.append(std)
    fig, axis = plt.subplots(figsize=(6.6, 4.4))
    axis.plot(epochs, means, "o-", lw=2.2, color="#8c3c86", label="Generator-only fine-tuning")
    axis.fill_between(epochs, [a-b for a,b in zip(means,stds)], [a+b for a,b in zip(means,stds)], color="#8c3c86", alpha=.18)
    axis.axhline(sum(classical) / len(classical), color="#333333", ls="--", label="Classical")
    axis.set(xlabel="Quantum fine-tuning epoch", ylabel="Fixed-noise test denoising MSE")
    axis.grid(True, ls="--", alpha=.4); axis.legend(); fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(os.path.join(output_dir, f"fig_d3_finetune_recovery.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_validation_abc(seeds=(0, 1, 2), epochs_classical=80, epochs_quantum=20,
                       timesteps=100, batch_size=128, k=16, classifier_epochs=5,
                       samples_per_digit=50):
    """Run the preregistered DiT Validation A--C protocol requested for the paper."""
    device = _device()
    output_dir = os.path.join(RESULT_DIR, "exp2_validation")
    os.makedirs(output_dir, exist_ok=True)
    train_loader, test_loader = mnist_loaders(batch_size=batch_size, pad=2, test_batch=256, drop_last=True, num_workers=0)
    diffusion = CosineDDPM(timesteps=timesteps, device=device)
    fixed_cases = make_fixed_test_cases(test_loader, timesteps=timesteps, repeats=4)
    print(f"Validation A--C on {device}: seeds={list(seeds)}, full MNIST test x 4 fixed noise/timestep pairs")
    torch.manual_seed(271828)
    classifier = train_classifier(train_loader, device, output_dir, epochs=classifier_epochs)
    classifier_accuracy = classifier_test_accuracy(classifier, test_loader, device)
    print(f"Independent classifier held-out accuracy={classifier_accuracy:.4f}")

    zero_rows, conditional_rows, finetune_rows, checkpoints = [], [], [], {}
    for seed in seeds:
        print(f"\n=== Classical seed {seed} ===")
        checkpoint = train_classical_checkpoint(seed, train_loader, diffusion, device, epochs_classical, output_dir)
        checkpoints[seed] = checkpoint
        classical = _model().to(device); classical.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        classical_mse = fixed_denoising_mse(classical, diffusion, fixed_cases, device)
        conditional_rows.append({"seed": seed, "phase": "classical",
                                 **conditional_metrics(classical, classifier, diffusion, device,
                                                       100000 + seed, samples_per_digit=samples_per_digit)})
        for gamma in (0.0, 0.1, 0.3, 1.0):
            quantum = load_quantum_from_checkpoint(checkpoint, device, gamma, k=k)
            mse = fixed_denoising_mse(quantum, diffusion, fixed_cases, device)
            zero_rows.append({"seed": seed, "gamma": gamma, "mse": mse, "classical_mse": classical_mse})
            if gamma in (0.1, 1.0):
                conditional_rows.append({"seed": seed, "phase": f"zero_gamma_{gamma:g}",
                                         **conditional_metrics(quantum, classifier, diffusion, device,
                                                               int(200000 + 100 * gamma) + seed,
                                                               samples_per_digit=samples_per_digit)})
        print(f"  seed={seed} classical fixed-MSE={classical_mse:.6f}")

    gamma1_mean = _mean_std([row["mse"] for row in zero_rows if row["gamma"] == 1.0])[0]
    classical_mean = _mean_std([row["classical_mse"] for row in zero_rows if row["gamma"] == 1.0])[0]
    fine_tune_gamma = 1.0 if gamma1_mean <= 1.10 * classical_mean else 0.1
    print(f"\nFine-tune rule: gamma=1 MSE={gamma1_mean:.6f}, classical={classical_mean:.6f}; selected gamma={fine_tune_gamma}")
    for seed in seeds:
        quantum = load_quantum_from_checkpoint(checkpoints[seed], device, fine_tune_gamma, k=k)
        curve = generator_finetune(quantum, train_loader, diffusion, fixed_cases, device, epochs_quantum)
        for row in curve:
            finetune_rows.append({"seed": seed, **row, "gamma": fine_tune_gamma})
        conditional_rows.append({"seed": seed, "phase": "fine_tuned",
                                 **conditional_metrics(quantum, classifier, diffusion, device,
                                                       300000 + seed, samples_per_digit=samples_per_digit)})

    _write_csv(os.path.join(output_dir, "zero_shot_coupling.csv"), zero_rows)
    _write_csv(os.path.join(output_dir, "conditional_accuracy.csv"), conditional_rows)
    _write_csv(os.path.join(output_dir, "finetune_recovery.csv"), finetune_rows)
    summary = {"seeds": list(seeds), "timesteps": timesteps, "test_noise_timestep_repeats": 4,
               "test_set": "full MNIST test set (10,000 images)", "gammas": [0.0, 0.1, 0.3, 1.0],
               "fine_tune_rule": "gamma=1 if its mean zero-shot MSE <= 1.10 x mean classical MSE; otherwise gamma=0.1",
               "selected_fine_tune_gamma": fine_tune_gamma,
               "conditional_samples_per_phase_per_seed": 10 * samples_per_digit,
               "classifier": "independently trained frozen MNIST ConvNet",
               "classifier_held_out_accuracy": classifier_accuracy,
               "conditional_samples_per_digit": samples_per_digit}
    with open(os.path.join(output_dir, "summary.json"), "w") as handle: json.dump(summary, handle, indent=2)
    _plot_results(zero_rows, conditional_rows, finetune_rows, output_dir)
    return summary


if __name__ == "__main__":
    run_validation_abc()
