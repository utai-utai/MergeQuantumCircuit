import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using Device: {DEVICE}")


# ==========================================
# 1. 几何引擎与量子核心层
# ==========================================
class GeometricQMLEngine:
    @staticmethod
    def extract_transfer_map(W, k):
        W_complex = W.to(torch.complex64)
        U, S, Vh = torch.linalg.svd(W_complex, full_matrices=False)
        Q = U[:, :k]
        A = Q.mH @ W_complex @ Q
        H = (A + A.mH) / 2.0
        return Q, H


class ClassicalResLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(dim))
        with torch.no_grad():
            self.weight.add_(torch.randn(dim, dim) * 0.01)

    def forward(self, x):
        return x + torch.matmul(x, self.weight.t())


class QuantumResLinearZeroShot(nn.Module):
    def __init__(self, classical_layer, k):
        super().__init__()
        W = classical_layer.weight.data.clone().cpu()
        Q, H = GeometricQMLEngine.extract_transfer_map(W, k)
        self.register_buffer('Q', Q.to(DEVICE))
        self.register_buffer('H', H.to(DEVICE))
        U = torch.linalg.matrix_exp(-1j * self.H)
        self.register_buffer('U_mH', U.mH)

    def forward(self, x):
        x_c = x.to(torch.complex64)
        x_sub = torch.matmul(x_c, self.Q)
        x_ev = torch.matmul(x_sub, self.U_mH)
        x_q = torch.matmul(x_ev, self.Q.mH).real
        return x + x_q


# ==========================================
# 2. MiniViT 架构
# ==========================================
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


def run_vit_experiment():
    print("\n" + "=" * 50)
    print(f"👁️ Experiment I: ViT Zero-Shot Classification")

    dim = 64
    k = 16
    epochs = 5

    os.makedirs('./data', exist_ok=True)
    os.makedirs('./result', exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(full_train, batch_size=128, shuffle=True)
    test_loader = DataLoader(test, batch_size=1000, shuffle=False)

    model = MiniViT(dim=dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    def evaluate(model_net):
        model_net.eval()
        correct = 0
        with torch.no_grad():
            for d, t in test_loader:
                d, t = d.to(DEVICE), t.to(DEVICE)
                pred = model_net(d).argmax(dim=1, keepdim=True)
                correct += pred.eq(t.view_as(pred)).sum().item()
        return 100. * correct / len(test_loader.dataset)

    results_log = []

    for ep in range(epochs):
        model.train()
        epoch_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()

            loss_ce = criterion(model(data), target)

            # 添加软微扰约束 (权重设为极小的 1e-4)
            loss_reg = 0.0
            for m in model.modules():
                if isinstance(m, ClassicalResLinear):
                    I = torch.eye(m.weight.shape[0], device=DEVICE)
                    loss_reg += torch.nn.functional.mse_loss(m.weight, I)

            loss = loss_ce + 1e-4 * loss_reg
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        acc_c = evaluate(model)

        classical_layer = model.block.core_layer
        model.block.core_layer = QuantumResLinearZeroShot(classical_layer, k).to(DEVICE)
        acc_q = evaluate(model)
        model.block.core_layer = classical_layer

        log_str = f"Epoch {ep + 1:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | [Classical] Acc: {acc_c:.2f}% | [Quantum k={k}] Acc: {acc_q:.2f}%"
        print("  " + log_str)
        results_log.append(log_str)

    with open('./result/vit_accuracy.txt', 'w', encoding='utf-8') as f:
        f.write("=== Experiment I: ViT Epoch-by-Epoch ===\n")
        f.write("\n".join(results_log))
    print("✅ ViT results saved to './result/vit_accuracy.txt'")


if __name__ == "__main__":
    t0 = time.time()
    run_vit_experiment()
    print(f"Time taken: {time.time() - t0:.2f}s")