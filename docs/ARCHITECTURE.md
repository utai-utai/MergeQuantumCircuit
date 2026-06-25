# 项目导读 — 一页看懂 MergeQuantumCircuit

> 目的:把经典 MLP 的**近恒等残差核 W** 解析地(零样本、量子端不训练)编译成
> **子空间量子酉算子**,并支持把两个这样的模型在李代数里**合并**。
> 目标会场 PRX Quantum。

---

## 1. 整个项目只有三层

```
① 引擎  src/geometric_qml.py   ← 全部理论都在这。167 行,是唯一"承重"的文件。
② 方法  src/simulation.py       ← 在 MNIST 上验证引擎(实验 A / B / C)
③ 实验  experiment_*.py          ← 4 个独立故事,都只是引擎的应用
```

**只需记住一个函数:** `transfer_map(W, k) -> (Q, U_A, H)`
其余一切都是围着它转。

---

## 2. 引擎在做什么(geometric_qml.py)

把一个近酉的方阵 `W` 变成能在 `log2(k)` 个量子比特上跑的酉算子:

```
W ──SVD取前k个左奇异向量──▶ Q          (n×k 标准正交 frame)
A = Qᴴ W Q                              (k×k 子空间块)
A ──polar分解──▶ U_A = 最近酉           (nearest_unitary)
U_A ──H = i·log(U_A)──▶ H               (厄米生成元,hermitian_generator)
混合算子  O = Q · e^{-iH} · Qᴴ          (mixing_operator)
```

**中心定理(误差分解,`error_decomposition`):**
```
‖W − O‖_F  ≤  ‖W − Π_Q(W)‖_F   +   ‖P_A − I‖_F
              └─ 截断误差 ─┘       └─ 非酉误差 ─┘
                                   精确 = √Σ(σⱼ(A)−1)²
```
**承重假设:W 必须"近酉"**(残差 + 单位阵初始化 → 非酉项才是二阶小量)。

**合并(`merge_generators`):** 覆盖 frame `Q_C = orth([Q_A, Q_B])`(维度 ≤ 2k)→
把两个生成元平行传输到 `Q_C` 上 → 平均。合并代价主项 ∝ ‖H_A′ − H_B′‖²。

---

## 3. 每个实验都是同一个套路

```
训一个带"近恒等残差核"的经典模型
        │   (核 = I + 小噪声;训练时加软正则 mse_loss(W, I) 维持近酉)
        ▼
调 transfer_map(W, k) 把核换成量子算子 O
        │   零样本(vit/hardware) 或 可训练(dit 把 H 设为参数再微调)
        ▼
评估,结果写进 result/
```

| 文件 | 故事 | 量子层是 | 产出(result/) |
|---|---|---|---|
| `simulation.py` | 方法验证 A零样本 / B误差分解 / C合并 | 零样本 | `sim_*.txt/png`, `sim_circuit.txt` |
| `experiment_vit.py` | ViT 残差核逐 epoch 零样本替换 | 零样本 | `vit_accuracy.png/txt` |
| `experiment_dit.py` | 扩散模型 经典→零样本→微调 三阶段 | **可训练** | `*_hd.png`, `dit_metrics.txt` |
| `experiment_hardware.py` | IBM 真机 Hellinger 保真度 vs k | 零样本 | `hw_ibm_kobe_*.json/png` |
| `experiment_bp.py` | 梯度方差 n=16→128(论文 Fig) | 仅参数化电路 | `bp_*.json`, `bp_gradient_variance.pdf` |

辅助:`hw_auth_check.py`(读 `.env` 连 IBM)、`replot_bp.py`(用已存 JSON 重画图,不重跑)。

---

## 4. 怎么跑

```bash
python -m src.simulation                 # 方法验证 A/B/C
python experiment_vit.py                 # 实验 I
python experiment_dit.py --smoke         # 实验 II(--smoke 是秒级冒烟测试)
python experiment_hardware.py            # 实验 III 本地 dry-run(--submit 上真机)
python experiment_bp.py                  # 实验 IV dry-run(--submit 上真机)
```

论文:`main.tex`(正文),`paper_build/` 里 `latexmk` 编译;图按裸文件名引用,
所以新图要先复制进 build 目录。

---

## 5. 已知的"冗杂"(将来若要精简,从这下手)

代码本身没坏,只是**跨文件重复**。本质代码就 `geometric_qml.py` 那 167 行;
其余文件各自重复了下面这些东西:

- **残差核** `ClassicalResLinear`:simulation / vit / dit 各写了一遍
- **量子替换层**:`QuantumHybridNet` / `QuantumResLinearZeroShot` /
  `TrainableQuantumResLinear` —— 3 个变体(注意:零样本 vs 可训练**确实不同**,
  不能粗暴合并)
- **脚手架**(`DEVICE`、`RESULT_DIR`、`makedirs`):5 个文件全有
- **MNIST 加载 / `evaluate()` / 软正则 `mse_loss(W,I)`**:各重复 2–3 份
- `_clean_unitary`(hardware)其实就是引擎里的 `nearest_unitary`

> 真要精简:把数据/路径/设备/残差核/train-eval 抽到 `src/common.py`,
> 各实验瘦身 import,约可砍 150–200 行;量子层因语义不同应保持独立。
> ⚠️ 这是论文复现代码,改实验脚本前需回归测试每个脚本仍跑出同样数字。
```
