"""
Experiment I (ViT) — zero-shot quantum transfer of the residual core inside a
Vision Transformer block, evaluated epoch-by-epoch on MNIST.

At every epoch the classical core is analytically replaced (no quantum-side
training) by the mixing operator O(Q,H)=Q e^{-iH} Q^H using the corrected
transfer map; the resulting hybrid model is evaluated.

  from src.experiments.exp1_vit import run
"""
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from src import RESULT_DIR, SEED
from src.core.data import mnist_loaders, near_identity_weight
from src.core.geometric_qml import transfer_map, mixing_operator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================================
# Classical residual core + zero-shot quantum replacement
# ==========================================================================
class ClassicalResLinear(nn.Module):
    """Residual near-identity core:  x -> x + x W^T,  W = I + small noise."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(near_identity_weight(dim))

    def forward(self, x):
        return x + torch.matmul(x, self.weight.t())


class QuantumResLinearZeroShot(nn.Module):
    """Replace the W-action by the mixing operator O = Q U_A Q^H (zero-shot)."""

    def __init__(self, classical_layer, k):
        super().__init__()
        W = classical_layer.weight.data.clone().cpu()
        Q, U_A, _ = transfer_map(W, k)
        O = mixing_operator(Q.to(DEVICE), U_A.to(DEVICE))
        self.register_buffer("O_T", O.T.contiguous())

    def forward(self, x):
        x_q = (x.to(torch.complex64) @ self.O_T).real
        return x + x_q


# ==========================================================================
# MiniViT
# ==========================================================================
class MiniViTBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc_act = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.core_layer = ClassicalResLinear(dim)

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        h = self.fc_act(self.norm2(x))
        x = x + self.core_layer(h)
        return x


class MiniViT(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.patch_embed = nn.Conv2d(1, dim, kernel_size=7, stride=7)
        self.pos_embed = nn.Parameter(torch.randn(1, 17, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.block = MiniViTBlock(dim, heads=4)
        self.head = nn.Linear(dim, 10)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.block(x)
        return self.head(x[:, 0])


def run(dim=64, k=16, epochs=5):
    print("\n" + "=" * 60 + "\nExperiment I: ViT zero-shot quantum transfer")
    torch.manual_seed(SEED)
    train_loader, test_loader = mnist_loaders(batch_size=128)

    model = MiniViT(dim=dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def evaluate(net):
        net.eval()
        correct = 0
        for d, t in test_loader:
            d, t = d.to(DEVICE), t.to(DEVICE)
            correct += (net(d).argmax(dim=1) == t).sum().item()
        return 100.0 * correct / len(test_loader.dataset)

    hist_c, hist_q, log_lines = [], [], []
    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for data_, target in train_loader:
            data_, target = data_.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(data_), target)
            I = torch.eye(dim, device=DEVICE)            # soft identity anchor (NU)
            loss = loss + 1e-4 * nn.functional.mse_loss(model.block.core_layer.weight, I)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        acc_c = evaluate(model)
        classical_layer = model.block.core_layer
        model.block.core_layer = QuantumResLinearZeroShot(classical_layer, k).to(DEVICE)
        acc_q = evaluate(model)
        model.block.core_layer = classical_layer

        hist_c.append(acc_c); hist_q.append(acc_q)
        line = (f"Epoch {ep:02d}/{epochs} | loss {epoch_loss / len(train_loader):.4f} | "
                f"[classical] {acc_c:.2f}% | [quantum k={k}] {acc_q:.2f}% | gap {acc_c - acc_q:+.2f}")
        print("  " + line); log_lines.append(line)

    _plot(hist_c, hist_q, k)


def _plot(hist_c, hist_q, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs_x = list(range(1, len(hist_c) + 1))
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(epochs_x, hist_c, "o-", lw=2.2, ms=6, color="#1f3b73", label="classical core")
    plt.plot(epochs_x, hist_q, "s--", lw=2.2, ms=6, color="#b22222",
             label=f"quantum zero-shot (k={k}, {int.bit_length(k) - 1} qubits)")
    plt.xlabel("epoch", fontsize=12)
    plt.ylabel("MNIST test accuracy (%)", fontsize=12)
    plt.title("ViT: zero-shot quantum transfer of the residual core", fontsize=13, pad=8)
    plt.xticks(epochs_x)
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    fig_path = os.path.join(RESULT_DIR, "exp1_accuracy.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("  saved ->", os.path.relpath(fig_path, os.path.dirname(RESULT_DIR)))
