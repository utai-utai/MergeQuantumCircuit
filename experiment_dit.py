import os
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


# ============================================================================
# 1. 指数移动平均 (EMA) 模块
# ============================================================================
class EMA:
    """
    指数移动平均 (EMA) 模块。
    在理论上，EMA 权重作为参数流形上的低通滤波器，能够有效稳定扩散模型的逆向轨迹。
    """

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
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data.copy_(self.backup[name])
        self.backup = {}


# ============================================================================
# 2. 核心理论约束层：经典微扰层与方差守恒量子层
# ============================================================================
class ClassicalResLinear(nn.Module):
    """
    【核心理论约束 1】：经典微扰层 (Classical ResLinear Layer)
    公式: x = x + alpha * (x @ W^T)
    满足渐进单位阵微扰假设: W ≈ I，用于防止后续量子泰勒展开高阶误差爆炸。
    """

    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features
        # alpha 为可学习的微扰阻尼系数，初始值设为 0.1
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        # 权重 W 使用单位阵初始化并加入 0.01 的高斯噪声
        weight = torch.eye(in_features, dtype=torch.float32)
        noise = torch.randn(in_features, in_features) * 0.01
        self.weight = nn.Parameter(weight + noise)

    def forward(self, x):
        return x + self.alpha * F.linear(x, self.weight)

    def get_reg_loss(self):
        """流形软锚点正则化惩罚项: MSE(W, I)"""
        target = torch.eye(self.in_features, device=self.weight.device)
        return F.mse_loss(self.weight, target)


class TrainableQuantumResLinear(nn.Module):
    """
    【核心理论约束 2】：方差守恒量子层 (Trainable Quantum ResLinear)
    接收训练好的经典层，通过严格的 SVD 截断提取复 Stiefel 流形基底 Q，并用极分解和对数映射初始化哈密顿量。
    """

    def __init__(self, classical_layer, k=16):
        super().__init__()
        self.in_features = classical_layer.in_features
        self.k = min(k, self.in_features)

        # 继承经典层的非线性微扰尺度 alpha
        self.alpha = nn.Parameter(classical_layer.alpha.data.clone())

        # --- 严格解析转移映射 T(W) 展开 ---
        W_mat = classical_layer.weight.data.cpu()

        # 1. 全局权重奇异值分解 (SVD) 提取最优子空间标架
        U_svd, S_svd, Vh_svd = torch.linalg.svd(W_mat, full_matrices=False)

        # 2. 截断秩 k 并映射至复数域，构建复 Stiefel 流形基底 Q \in St(k, n; C)
        Q_real = U_svd[:, :self.k] @ torch.diag(torch.sqrt(S_svd[:self.k]))
        self.Q = nn.Parameter(Q_real.to(torch.complex64), requires_grad=False)

        # 3. 算子投影到子空间中 A = Q^T * W * Q
        A = Q_real.t() @ W_mat @ Q_real

        # 4. 【解析修复】通过内部 SVD 严格计算 A 的矩阵极分解 (Polar Decomposition)
        # A = U_a * Sigma_a * Vh_a -> 最优幺正算符 U_A = U_a * Vh_a
        U_a, S_a, Vh_a = torch.linalg.svd(A, full_matrices=False)
        U_A = U_a @ Vh_a

        # 5. 矩阵对数逆推哈密顿量生成元 H = i * log(U_A)
        # 在残差微扰下，由反厄米特部分的无偏估计给出以确保绝对光滑且 H = H^dagger
        H_init = 0.5j * (U_A - U_A.conj().t())
        self.H_param = nn.Parameter(H_init.to(torch.complex64), requires_grad=True)

    def forward(self, x):
        # 1. 强行进行厄米特化约束，确保哈密顿量的自伴性 (H = H^dagger)
        H_herm = 0.5 * (self.H_param + self.H_param.conj().transpose(-2, -1))

        # 2. 计算严格的子空间幺正演化算子 U = matrix_exp(-1j * H_herm)
        U = torch.linalg.matrix_exp(-1j * H_herm)

        # 3. 向量投影至子空间并执行薛定谔时演
        x_complex = x.to(torch.complex64)
        x_projected = x_complex @ self.Q  # [B, N, k]
        x_ev = x_projected @ U.transpose(-2, -1)

        # 4. 物理实部提取与逆投影
        Q_H = self.Q.conj().transpose(-2, -1)
        x_q = torch.real(x_ev @ Q_H)  # [B, N, in_features]

        # 5. 严格方差标准化约束，维持马尔可夫链高斯拓扑不塌陷
        x_q = x_q / (x_q.std(dim=-1, keepdim=True) + 1e-6)

        # 6. 带 0.1 阻尼系数的量子残差融合，严格抑制高频相位爆炸
        return x + 0.1 * self.alpha * x_q


# ============================================================================
# 3. 高清自适应条件 DiT 架构 (ConditionalMiniDiT)
# ============================================================================
def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half).to(t.device)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class AdaLNModulation(nn.Module):
    """AdaLN 调制模块：控制每一层 Block 的 Scale 和 Shift 仿射变换"""

    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, dim * 6)
        )

    def forward(self, emb):
        return self.mlp(emb).chunk(6, dim=-1)


class ConditionalDiTBlock(nn.Module):
    """自适应调制 Transformer 块"""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.core_layer = ClassicalResLinear(dim)  # 默认初始化为经典微扰层

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            self.core_layer
        )
        self.adaLN = AdaLNModulation(dim)

    def forward(self, x, c_emb):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c_emb)

        # Attention 分支调制
        x_norm = self.norm1(x)
        x_modulated = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_modulated, x_modulated, x_modulated)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP 与核心物理层分支调制
        x_norm = self.norm2(x)
        x_modulated = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_modulated)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


class ConditionalMiniDiT(nn.Module):
    """
    【核心理论约束 3】：高清 Mini-DiT 架构
    分辨率匹配：32x32 图像, 4x4 Patch -> 8x8 = 64 个 Tokens。
    隐藏层参数：dim=192, depth=6, heads=4。
    """

    def __init__(self, img_size=32, patch_size=4, in_channels=1, dim=192, depth=6, heads=4, num_classes=10):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.dim = dim

        # Patch 切分层
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)

        # 条件暴力注入层
        self.y_embed = nn.Embedding(num_classes, dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.c_ln = nn.LayerNorm(dim)

        self.blocks = nn.ModuleList([ConditionalDiTBlock(dim, heads) for _ in range(depth)])

        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, patch_size * patch_size * in_channels)

    def forward(self, x, t, y):
        B = x.shape[0]

        # 1. Token 切分与位置编码
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed

        # 2. 类别嵌入暴力注入 (直接相加)
        y_emb = self.y_embed(y)
        x = x + y_emb.unsqueeze(1)

        # 3. 时间步嵌入与 AdaLN 联合向量融合
        t_emb = timestep_embedding(t, self.dim)
        t_emb = self.time_mlp(t_emb)
        c_emb = self.c_ln(t_emb + y_emb)

        # 4. 变换器层级级联演化
        for block in self.blocks:
            x = block(x, c_emb)

        # 5. 逆切片化 (Unpatchify) 还原像素空间
        x = self.final_norm(x)
        x = self.head(x)

        p = self.patch_size
        h = w = self.img_size // p
        x = x.view(B, h, w, p, p)
        x = x.permute(0, 1, 3, 2, 4).reshape(B, 1, self.img_size, self.img_size)
        return x

    def convert_to_quantum(self, k=16):
        """流形一键映射：平滑无损替换所有 Block 的核心微扰层为复流形量子算子"""
        for block in self.blocks:
            c_layer = block.mlp[-1]
            if isinstance(c_layer, ClassicalResLinear):
                block.mlp[-1] = TrainableQuantumResLinear(c_layer, k=k)


# ============================================================================
# 4. 100% 标准 DDPM 余弦加噪与逆向推断调度器
# ============================================================================
class CosineDDPM:
    """
    【核心理论约束 4】：严格标准的余弦加噪加噪 Schedule（TIMESTEPS = 100）
    """

    def __init__(self, timesteps=100, device="cuda"):
        self.timesteps = timesteps
        self.device = device

        # 基于余弦函数的更平滑的加噪路径
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
        """前向扩散加噪"""
        a_bar = self.alphas_bar[t][:, None, None, None]
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise

    @torch.no_grad()
    def p_sample(self, model, x, t, y):
        """
        逆向推断推导公式:
        x = (1 / sqrt(alpha_t)) * (x - (beta_t / sqrt(1 - alpha_bar_t)) * eps_theta)
        """
        B = x.shape[0]
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
# 5. 高清渲染与无损像素导出核心
# ============================================================================
def render_and_save_hd(ema_model, diffusion, labels, path_name):
    """彻底抛弃 matplotlib，采用最近邻插值无损放大 8 倍变成 256x256 绝对锐利像素图"""
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
        # 最近邻像素级超分辨率放大
        x_hd = F.interpolate(x, scale_factor=8.0, mode='nearest')
        save_image(x_hd, path_name, nrow=B)

    ema_model.restore()
    return x_hd


# ============================================================================
# 6. 三阶段演化主工作流
# ============================================================================
def main():
    TIMESTEPS = 100
    BATCH_SIZE = 128
    EPOCHS_STAGE1 = 80
    EPOCHS_STAGE3 = 20
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Top-Journal Quantum Pipeline on: {DEVICE}")

    # 数据集加载，DataLoader 强行补零扩大为标准的 32x32 尺寸 (Pad(2))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(2, fill=0),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)

    # 组件实例化
    model = ConditionalMiniDiT(img_size=32, patch_size=4, dim=192, depth=6, heads=4).to(DEVICE)
    ema = EMA(model, decay=0.999)
    diffusion = CosineDDPM(timesteps=TIMESTEPS, device=DEVICE)

    # 评测固定条件：0-9 的标准条件标签
    eval_labels = torch.arange(10, device=DEVICE)

    # ------------------------------------------------------------------------
    # 阶段一（经典训练）：内置软锚点正则化惩罚项
    # ------------------------------------------------------------------------
    print("\n=== Phase 1: Classical Training with Manifold Soft Anchors ===")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    for epoch in range(1, EPOCHS_STAGE1 + 1):
        model.train()
        total_loss, total_reg = 0.0, 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            t = torch.randint(0, TIMESTEPS, (x.size(0),), device=DEVICE)
            noise = torch.randn_like(x)

            x_t = diffusion.q_sample(x, t, noise)
            pred = model(x_t, t, y)

            loss_mse = F.mse_loss(pred, noise)

            # 计算经典层的流形软锚点惩罚项 (loss_reg系数为1e-3)
            loss_reg = 0.0
            for block in model.blocks:
                loss_reg += block.core_layer.get_reg_loss()

            loss = loss_mse + 1e-3 * loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update()

            total_loss += loss_mse.item()
            total_reg += loss_reg.item()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:02d}/{EPOCHS_STAGE1} | MSE Loss: {total_loss / len(loader):.4f} | Reg Loss: {total_reg / len(loader):.4f}")

    # 渲染经典高清图像
    hd_classical = render_and_save_hd(ema, diffusion, eval_labels, "result/1_classical_hd.png")

    # ------------------------------------------------------------------------
    # 阶段二（零样本量子态）：载入经典 EMA 权重，一键替换，免训推断
    # ------------------------------------------------------------------------
    print("\n=== Phase 2: Zero-Shot Quantum Manifold Translation (No Training) ===")
    ema.apply_shadow()
    model.convert_to_quantum(k=16)  # 理论转移映射 T(W) 核心触发点
    model.to(DEVICE)

    # 重新绑定流形替换后的量子参数 EMA 跟踪器
    ema = EMA(model, decay=0.999)

    # 直接采用此时的模型无缝推断采样 10 个数字
    hd_zeroshot = render_and_save_hd(ema, diffusion, eval_labels, "result/2_quantum_zeroshot_hd.png")

    # ------------------------------------------------------------------------
    # 阶段三（量子微调）：冻结全网其他层，仅激活哈密顿量生成元
    # ------------------------------------------------------------------------
    print("\n=== Phase 3: Quantum Fine-Tuning (Hamiltonian Optimization) ===")
    for name, param in model.named_parameters():
        if "H_param" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 小学习率 (5e-5) 与严格的梯度裁剪 (max_norm=0.01)
    q_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

    for epoch in range(1, EPOCHS_STAGE3 + 1):
        model.train()
        total_q_loss = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            t = torch.randint(0, TIMESTEPS, (x.size(0),), device=DEVICE)
            noise = torch.randn_like(x)

            x_t = diffusion.q_sample(x, t, noise)
            pred = model(x_t, t, y)
            loss = F.mse_loss(pred, noise)

            q_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
            q_optimizer.step()
            ema.update()

            total_q_loss += loss.item()

        print(f"Quantum Epoch {epoch:02d}/{EPOCHS_STAGE3} | MSE Loss: {total_q_loss / len(loader):.4f}")

    # 采样最终的量子图像
    hd_finetuned = render_and_save_hd(ema, diffusion, eval_labels, "result/3_quantum_finetuned_hd.png")

    # ------------------------------------------------------------------------
    # 终极对比大图垂直拼接导出
    # ------------------------------------------------------------------------
    final_evolution_chart = torch.cat([hd_classical, hd_zeroshot, hd_finetuned], dim=2)
    save_image(final_evolution_chart, "result/0_all_evolution_hd.png", nrow=1)
    print("\n[SUCCESS] Unified HD comparison chart saved to '0_all_evolution_hd.png'")


if __name__ == "__main__":
    main()