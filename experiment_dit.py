"""
EXPERIMENT II (DiT) — class-conditional diffusion (DDPM) on MNIST whose core
residual MLP layers are transferred to trainable quantum mixing operators.

Three phases (docs/THEORY.md §3, §4, §6):
  Phase 1 — classical training (soft identity anchor keeps cores near-unitary, NU).
  Phase 2 — zero-shot quantum transfer of every core via the corrected map
            T(W) = (Q, U_A, H),  H = i log U_A  (no quantum-side training).
  Phase 3 — few-shot quantum fine-tuning: freeze everything except the Hermitian
            generators H (the warm-started, identity-neighbourhood initialisation
            of §6), optimise them with a small step / clipped gradients.

Artefacts (HD sample grids + a metrics log) are written to ./result.
"""

import os
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from src.geometric_qml import transfer_map

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(_HERE, "result")
DATA_DIR = os.path.join(_HERE, "data")
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================================
# 1. EMA
# ============================================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = ((1.0 - self.decay) * param.data
                                     + self.decay * self.shadow[name]).clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ============================================================================
# 2. Classical residual core + trainable quantum core
# ============================================================================
class ClassicalResLinear(nn.Module):
    """x -> x + alpha * x W^T,  W = I + small noise  (near-identity, NU)."""

    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        weight = torch.eye(in_features, dtype=torch.float32) + torch.randn(in_features, in_features) * 0.01
        self.weight = nn.Parameter(weight)

    def forward(self, x):
        return x + self.alpha * F.linear(x, self.weight)

    def get_reg_loss(self):
        target = torch.eye(self.in_features, device=self.weight.device)
        return F.mse_loss(self.weight, target)


class TrainableQuantumResLinear(nn.Module):
    """Replace the W-action by O = Q e^{-iH} Q^H with a TRAINABLE Hermitian
    generator H (warm-started from the analytic transfer map; §6).

    Corrections vs. the original code:
      * Q is a genuine orthonormal Stiefel frame (Q^H Q = I_k) — the previous
        `U_svd[:,:k] @ diag(sqrt(S))` rescaled the columns and broke isometry.
      * H is initialised from the polar/log transfer map H = i log U_A.
    The std-normalisation + alpha damping in the forward pass are diffusion
    variance-preservation heuristics, kept for sampling stability.
    """

    def __init__(self, classical_layer, k=16):
        super().__init__()
        self.in_features = classical_layer.in_features
        self.k = min(k, self.in_features)
        self.alpha = nn.Parameter(classical_layer.alpha.data.clone())

        W = classical_layer.weight.data.cpu()
        Q, _, H = transfer_map(W, self.k)              # orthonormal Q, H = i log U_A
        self.register_buffer("Q", Q)                   # (n, k) complex, Q^H Q = I
        self.H_param = nn.Parameter(H)                 # (k, k) Hermitian, trainable

    def forward(self, x):
        H = 0.5 * (self.H_param + self.H_param.mH)      # enforce Hermiticity
        U = torch.linalg.matrix_exp(-1j * H)
        xc = x.to(torch.complex64)
        x_sub = xc @ self.Q                             # project onto subspace
        x_ev = x_sub @ U.transpose(-2, -1)              # unitary evolution
        x_q = torch.real(x_ev @ self.Q.mH)              # back-project, take real part
        x_q = x_q / (x_q.std(dim=-1, keepdim=True) + 1e-6)
        return x + 0.1 * self.alpha * x_q


# ============================================================================
# 3. Conditional Mini-DiT
# ============================================================================
def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half).to(t.device)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class AdaLNModulation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(dim, dim * 6))

    def forward(self, emb):
        return self.mlp(emb).chunk(6, dim=-1)


class ConditionalDiTBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.core_layer = ClassicalResLinear(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            self.core_layer,
        )
        self.adaLN = AdaLNModulation(dim)

    def forward(self, x, c_emb):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c_emb)
        x_norm = self.norm1(x)
        x_modulated = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_modulated, x_modulated, x_modulated)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x_norm = self.norm2(x)
        x_modulated = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_modulated)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


class ConditionalMiniDiT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=1, dim=192, depth=6, heads=4, num_classes=10):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.dim = dim
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)
        self.y_embed = nn.Embedding(num_classes, dim)
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.c_ln = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([ConditionalDiTBlock(dim, heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, patch_size * patch_size * in_channels)

    def forward(self, x, t, y):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        y_emb = self.y_embed(y)
        x = x + y_emb.unsqueeze(1)
        t_emb = self.time_mlp(timestep_embedding(t, self.dim))
        c_emb = self.c_ln(t_emb + y_emb)
        for block in self.blocks:
            x = block(x, c_emb)
        x = self.head(self.final_norm(x))
        p = self.patch_size
        h = w = self.img_size // p
        x = x.view(B, h, w, p, p).permute(0, 1, 3, 2, 4).reshape(B, 1, self.img_size, self.img_size)
        return x

    def convert_to_quantum(self, k=16):
        for block in self.blocks:
            c_layer = block.mlp[-1]
            if isinstance(c_layer, ClassicalResLinear):
                block.mlp[-1] = TrainableQuantumResLinear(c_layer, k=k)


# ============================================================================
# 4. Cosine DDPM
# ============================================================================
class CosineDDPM:
    def __init__(self, timesteps=100, device="cuda"):
        self.timesteps = timesteps
        self.device = device
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_bar = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        self.betas = torch.clamp(betas, min=1e-4, max=0.999).to(device)
        self.alphas = (1.0 - self.betas).to(device)
        self.alphas_bar = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)

    def q_sample(self, x0, t, noise):
        a_bar = self.alphas_bar[t][:, None, None, None]
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise

    @torch.no_grad()
    def p_sample(self, model, x, t, y):
        beta_t = self.betas[t][:, None, None, None]
        sqrt_alpha_t = self.sqrt_alphas[t][:, None, None, None]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_bar[t][:, None, None, None]
        eps_theta = model(x, t, y)
        mean = (1.0 / sqrt_alpha_t) * (x - (beta_t / sqrt_one_minus_alpha_bar_t) * eps_theta)
        if t[0] > 0:
            noise = torch.randn_like(x)
            alphas_bar_prev = self.alphas_bar[t - 1][:, None, None, None]
            alphas_bar_t = self.alphas_bar[t][:, None, None, None]
            variance = ((1.0 - alphas_bar_prev) / (1.0 - alphas_bar_t)) * beta_t
            return mean + torch.sqrt(variance) * noise
        return mean


# ============================================================================
# 5. HD rendering
# ============================================================================
def render_and_save_hd(ema_model, diffusion, labels, path_name):
    ema_model.apply_shadow()
    ema_model.model.eval()
    device = next(ema_model.model.parameters()).device
    B = labels.shape[0]
    with torch.no_grad():
        x = torch.randn(B, 1, 32, 32, device=device)
        for t in reversed(range(diffusion.timesteps)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            x = diffusion.p_sample(ema_model.model, x, t_tensor, labels)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x_hd = F.interpolate(x, scale_factor=8.0, mode="nearest")
        save_image(x_hd, path_name, nrow=B)
    ema_model.restore()
    return x_hd


# ============================================================================
# 6. Pipeline
# ============================================================================
def main(epochs_stage1=80, epochs_stage3=20, timesteps=100, batch_size=128, k=16):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing quantum diffusion pipeline on: {DEVICE} "
          f"(stage1={epochs_stage1}, stage3={epochs_stage3})")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(2, fill=0),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)

    model = ConditionalMiniDiT(img_size=32, patch_size=4, dim=192, depth=6, heads=4).to(DEVICE)
    ema = EMA(model, decay=0.999)
    diffusion = CosineDDPM(timesteps=timesteps, device=DEVICE)
    eval_labels = torch.arange(10, device=DEVICE)
    metrics = []

    # ---- Phase 1: classical training ----
    print("\n=== Phase 1: Classical training (manifold soft anchors) ===")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    for epoch in range(1, epochs_stage1 + 1):
        model.train()
        total_loss, total_reg = 0.0, 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            t = torch.randint(0, timesteps, (x.size(0),), device=DEVICE)
            noise = torch.randn_like(x)
            pred = model(diffusion.q_sample(x, t, noise), t, y)
            loss_mse = F.mse_loss(pred, noise)
            loss_reg = sum(b.core_layer.get_reg_loss() for b in model.blocks)
            loss = loss_mse + 1e-3 * loss_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update()
            total_loss += loss_mse.item()
            total_reg += float(loss_reg.detach())
        if epoch % 10 == 0 or epoch == 1:
            line = (f"[Phase1] Epoch {epoch:02d}/{epochs_stage1} | MSE {total_loss / len(loader):.4f} "
                    f"| Reg {total_reg / len(loader):.4f}")
            print(line); metrics.append(line)
    hd_classical = render_and_save_hd(ema, diffusion, eval_labels, os.path.join(RESULT_DIR, "1_classical_hd.png"))

    # ---- Phase 2: zero-shot quantum transfer ----
    print("\n=== Phase 2: Zero-shot quantum transfer (no training) ===")
    ema.apply_shadow()
    model.convert_to_quantum(k=k)
    model.to(DEVICE)
    ema = EMA(model, decay=0.999)
    hd_zeroshot = render_and_save_hd(ema, diffusion, eval_labels, os.path.join(RESULT_DIR, "2_quantum_zeroshot_hd.png"))
    metrics.append(f"[Phase2] converted cores to quantum (k={k}, {int.bit_length(k) - 1} qubits), zero-shot sampled")

    # ---- Phase 3: quantum fine-tuning (Hamiltonian only) ----
    print("\n=== Phase 3: Quantum fine-tuning (Hamiltonian generators only) ===")
    for name, param in model.named_parameters():
        param.requires_grad = ("H_param" in name)
    q_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
    for epoch in range(1, epochs_stage3 + 1):
        model.train()
        total_q_loss = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            t = torch.randint(0, timesteps, (x.size(0),), device=DEVICE)
            noise = torch.randn_like(x)
            pred = model(diffusion.q_sample(x, t, noise), t, y)
            loss = F.mse_loss(pred, noise)
            q_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
            q_optimizer.step()
            ema.update()
            total_q_loss += loss.item()
        line = f"[Phase3] Quantum Epoch {epoch:02d}/{epochs_stage3} | MSE {total_q_loss / len(loader):.4f}"
        print(line); metrics.append(line)
    hd_finetuned = render_and_save_hd(ema, diffusion, eval_labels, os.path.join(RESULT_DIR, "3_quantum_finetuned_hd.png"))

    # ---- combined evolution chart ----
    chart = torch.cat([hd_classical, hd_zeroshot, hd_finetuned], dim=2)
    save_image(chart, os.path.join(RESULT_DIR, "0_all_evolution_hd.png"), nrow=1)
    with open(os.path.join(RESULT_DIR, "dit_metrics.txt"), "w", encoding="utf-8") as f:
        f.write("=== Experiment II: DiT quantum diffusion ===\n")
        f.write("rows of 0_all_evolution_hd.png: classical | zero-shot quantum | fine-tuned quantum\n")
        f.write("\n".join(metrics) + "\n")
    print("\n[SUCCESS] HD comparison + dit_metrics.txt saved to result/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", type=int, default=80, help="classical training epochs")
    parser.add_argument("--stage3", type=int, default=20, help="quantum fine-tuning epochs")
    parser.add_argument("--smoke", action="store_true", help="tiny run to verify end-to-end")
    parser.add_argument("-k", type=int, default=16)
    args = parser.parse_args()
    torch.manual_seed(0)
    if args.smoke:
        main(epochs_stage1=2, epochs_stage3=1, k=args.k)
    else:
        main(epochs_stage1=args.stage1, epochs_stage3=args.stage3, k=args.k)
