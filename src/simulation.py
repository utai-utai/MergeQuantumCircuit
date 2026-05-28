import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import pennylane as qml

# 🚀 自动创建 result 文件夹用于保存论文图表
os.makedirs('../result', exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using Device: {DEVICE}")

# ==========================================
# 0. 物理量子电路工厂 (Quantum Hardware Factory)
# ==========================================
_qnodes = {}


def get_qnode(k, device_name="default.qubit"):
    """
    根据子空间截断秩 k，动态分配对应的量子比特数。
    例如 k=16，物理上只需要 log2(16) = 4 个量子比特！
    """
    if k not in _qnodes:
        num_qubits = int(math.log2(k))
        dev = qml.device(device_name, wires=num_qubits)

        @qml.qnode(dev, interface="torch")
        def stiefel_quantum_circuit(state, U_matrix):
            # Step 1: 振幅编码 (Amplitude Encoding)
            qml.StatePrep(state, wires=range(num_qubits))

            # Step 2: 酉演化 (Unitary Synthesis)
            qml.QubitUnitary(U_matrix, wires=range(num_qubits))

            # Step 3: 态读出 (State Readout)
            return qml.state()

        _qnodes[k] = stiefel_quantum_circuit
    return _qnodes[k]


# ==========================================
# 1. 纯几何理论引擎 (Geometric QML Engine)
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

    @staticmethod
    def connection_induced_transport(Q_A, H_A, Q_B, H_B, Q_C):
        """定理 3：基于正规联络的流形平行传输 (Lift-Project-Restrict)
           完美免疫 SVD 的相位规范模糊性，且保持物理量子比特数不变。"""
        # 1. 提升 (Lift)：Q_A @ H_A @ Q_A.mH 形成全局无歧义的哈密顿量
        # 2. 投影与限制 (Project & Restrict)：两侧乘 Q_C 投影到公共子空间
        H_A_prime = Q_C.mH @ (Q_A @ H_A @ Q_A.mH) @ Q_C
        H_B_prime = Q_C.mH @ (Q_B @ H_B @ Q_B.mH) @ Q_C
        return H_A_prime, H_B_prime


# ==========================================
# 2. 经典模型 (Classical ResNet)
# ==========================================
class ClassicalResNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super(ClassicalResNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc_in = nn.Linear(28 * 28, hidden_dim)
        self.core_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.eye_(self.core_layer.weight)
        with torch.no_grad():
            self.core_layer.weight.add_(torch.randn_like(self.core_layer.weight) * 0.01)
        self.fc_out = nn.Linear(hidden_dim, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc_in(x))
        x = x + self.core_layer(x)
        x = self.relu(x)
        x = self.fc_out(x)
        return x


# ==========================================
# 3. 升级版：真·量子混合网络 (Physical Edition)
# ==========================================
class QuantumHybridNet(nn.Module):
    def __init__(self, classical_model, Q, H):
        super(QuantumHybridNet, self).__init__()
        self.flatten = classical_model.flatten
        self.fc_in = classical_model.fc_in
        self.fc_out = classical_model.fc_out
        self.relu = classical_model.relu

        self.register_buffer('Q', Q)
        self.register_buffer('H', H)

        U_matrix = torch.linalg.matrix_exp(-1j * H)
        self.register_buffer('U_circuit', torch.conj(U_matrix))

        self.k = Q.shape[1]
        self.qnode = get_qnode(self.k)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc_in(x))

        x_complex = x.to(torch.complex64)
        x_subspace = x_complex @ self.Q

        batch_size = x_subspace.shape[0]
        quantum_outputs = []

        norms = torch.linalg.norm(x_subspace, dim=1, keepdim=True) + 1e-12
        normalized_states = x_subspace / norms

        for i in range(batch_size):
            state_in = normalized_states[i]
            quantum_out = self.qnode(state_in, self.U_circuit)
            quantum_outputs.append(quantum_out)

        x_evolved = torch.stack(quantum_outputs).to(x.device)
        x_evolved = x_evolved * norms

        x_quantum_core = (x_evolved @ self.Q.mH).real

        x = x + x_quantum_core
        x = self.relu(x)
        x = self.fc_out(x)
        return x


# ==========================================
# 4. 数据与训练辅助函数
# ==========================================
def get_dataloaders():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    full_train = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_loader = DataLoader(datasets.MNIST('../data', train=False, transform=transform), batch_size=1000, shuffle=False)
    idx_0_4 = [i for i, (_, label) in enumerate(full_train) if label <= 4]
    idx_5_9 = [i for i, (_, label) in enumerate(full_train) if label >= 5]
    train_loader_all = DataLoader(full_train, batch_size=256, shuffle=True)
    train_loader_0_4 = DataLoader(Subset(full_train, idx_0_4), batch_size=256, shuffle=True)
    train_loader_5_9 = DataLoader(Subset(full_train, idx_5_9), batch_size=256, shuffle=True)
    return train_loader_all, train_loader_0_4, train_loader_5_9, test_loader


def train_model(model, train_loader, epochs=3, freeze_surrounding=False):
    model.to(DEVICE)
    if freeze_surrounding:
        for name, param in model.named_parameters():
            if 'core_layer' not in name:
                param.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
    return model


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            pred = model(data).argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    return 100. * correct / total


# ==========================================
# 实验执行模块 (全量测试版)
# ==========================================
def experiment_A(train_loader, test_loader, hidden_dim=64, k=16):
    print("\n" + "=" * 50)
    print(f"📊 Experiment A: Physical Quantum Transfer (k={k}, Full Test Set)")

    model = ClassicalResNet(hidden_dim=hidden_dim)
    # 训练 3 个 Epoch，给论文提供一个扎实的 95%+ Baseline
    model = train_model(model, train_loader, epochs=3)
    acc_c = evaluate(model, test_loader)

    W_trained = model.core_layer.weight.data.clone().cpu()
    Q, H = GeometricQMLEngine.extract_transfer_map(W_trained, k)

    q_model = QuantumHybridNet(model, Q.to(DEVICE), H.to(DEVICE)).to(DEVICE)

    print("\n[Circuit Compilation] 物理设备上的 4 Qubits 编译结构：")
    dummy_state = torch.nn.functional.normalize(torch.rand(k, dtype=torch.complex64), p=2, dim=0)
    print(qml.draw(q_model.qnode)(dummy_state, q_model.U_circuit))
    print("-" * 50)

    # 运行完整的 10000 张测试集图片
    print(
        f"Running Quantum Simulator on all {len(test_loader.dataset)} test samples... (This may take a minute on GPU)")
    acc_q = evaluate(q_model, test_loader)

    print(f"[Classical Baseline] Accuracy: {acc_c:.2f}%")
    print(f"[True Quantum Simulator] Accuracy: {acc_q:.2f}%")
    return W_trained


def experiment_B(W_trained, hidden_dim=64):
    print("\n" + "=" * 50)
    print("📈 Experiment B: Error Scaling vs. Subspace Rank k")
    k_list = [2, 4, 8, 16, 32, 64]
    errors = []
    W_complex = W_trained.to(torch.complex64)
    for k in k_list:
        Q, H = GeometricQMLEngine.extract_transfer_map(W_trained, k)
        U = torch.linalg.matrix_exp(-1j * H)
        O_matrix = Q @ U @ Q.mH
        err = torch.norm(W_complex - O_matrix, p='fro').item()
        errors.append(err)
        print(f"Rank k={k:<2d} | Frobenius Error ||W - O(Q,H)||_F = {err:.4f}")

    # 🌟 生成用于发表的高清图表并保存到 result 文件夹
    plt.figure(figsize=(8, 5))
    plt.plot(k_list, errors, marker='s', markersize=8, linewidth=2.5, color='#B22222')
    plt.title("Approximation Error vs. Subspace Dimension ($k$)", fontsize=16, pad=15)
    plt.xlabel(r"Truncation Rank $k$ (Log Scale 2^n)", fontsize=14)
    plt.ylabel(r"Frobenius Error $||W - \mathcal{O}(Q,H)||_F$", fontsize=14)

    # 增加网格和对数刻度以增加科研感
    plt.xscale('log', base=2)
    plt.xticks(k_list, labels=[str(k) for k in k_list])
    plt.grid(True, which="both", ls="--", alpha=0.6)

    # 标注出相位隆起区域
    plt.annotate('Phase Penalty\n(Non-unitary mismatch)',
                 xy=(8, max(errors)), xytext=(8, max(errors) + 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                 fontsize=12, ha='center')

    plt.tight_layout()
    save_path = os.path.join('../result', 'experiment_B_scaling.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Figure saved successfully to: {save_path}")


def experiment_C(train_loader_all, train_loader_0_4, train_loader_5_9, test_loader, hidden_dim=64, k=16):
    print("\n" + "=" * 50)
    print(f"🔗 Experiment C: Quantum Manifold Merging (k={k}, Full Test Set)")

    base_model = ClassicalResNet(hidden_dim=hidden_dim)
    base_model = train_model(base_model, train_loader_all, epochs=3)

    import copy
    model_A = copy.deepcopy(base_model)
    model_B = copy.deepcopy(base_model)
    model_A = train_model(model_A, train_loader_0_4, epochs=1, freeze_surrounding=True)
    model_B = train_model(model_B, train_loader_5_9, epochs=1, freeze_surrounding=True)

    W_A = model_A.core_layer.weight.data.clone().cpu()
    W_B = model_B.core_layer.weight.data.clone().cpu()

    model_C_classic = copy.deepcopy(base_model)
    W_C_target = 0.5 * W_A + 0.5 * W_B
    model_C_classic.core_layer.weight.data = W_C_target.to(DEVICE)
    acc_classic_merge = evaluate(model_C_classic, test_loader)

    # 🌟 修复核心：提取目标流形的公共覆盖标架 Q_C (严格保持 k 维)
    Q_C, _ = GeometricQMLEngine.extract_transfer_map(W_C_target, k)

    Q_A, H_A = GeometricQMLEngine.extract_transfer_map(W_A, k)
    Q_B, H_B = GeometricQMLEngine.extract_transfer_map(W_B, k)

    # 🌟 修复核心：使用基于 Q_C 的严格正规联络传输
    H_A_prime, H_B_prime = GeometricQMLEngine.connection_induced_transport(Q_A, H_A, Q_B, H_B, Q_C)

    # 在李代数切空间内完成量子合并
    H_C_merge = 0.5 * H_A_prime + 0.5 * H_B_prime
    quantum_merged_model = QuantumHybridNet(base_model, Q_C.to(DEVICE), H_C_merge.to(DEVICE)).to(DEVICE)

    print(f"Running Quantum Merging Simulator on all {len(test_loader.dataset)} test samples...")
    acc_quantum_merge = evaluate(quantum_merged_model, test_loader)

    print(f"[Classical Direct Averaging] Accuracy: {acc_classic_merge:.2f}%")
    print(f"[True Quantum Simulator]     Accuracy: {acc_quantum_merge:.2f}%")


# if __name__ == "__main__":
#     t0 = time.time()
#     train_all, train_0_4, train_5_9, test = get_dataloaders()
#
#     W_trained = experiment_A(train_all, test, hidden_dim=64, k=16)
#     experiment_B(W_trained, hidden_dim=64)
#     experiment_C(train_all, train_0_4, train_5_9, test, hidden_dim=64, k=16)
#
#     print(f"\n✅ All Physical Experiments Completed in {time.time() - t0:.2f} seconds.")
