# 修正版理论：近酉线性核的子空间量子化与生成元合并
## Corrected Theory — Subspace Quantization of Near-Unitary Linear Cores and Generator-Space Merging

> **本文地位**：这是把 `introduction.ipynb` / `merge_theory.ipynb` / `simulation_theory.ipynb` 中的错误命题修正、并补全所有隐藏假设后的**自洽版本**，可直接作为论文理论部分的骨架。
> 与 `THEORY.md` 的关系：`THEORY.md` 说明"哪里错、为什么错"；**本文是"改完后的正确版本"**。每节末尾的 `↩` 标注它替换了原 notebook 的哪条命题。

---

## 0. 适用范围与核心假设（先把前提钉死）

本理论**只**声称对满足下述假设的线性核成立。所有"零训练/少训练"的结论都依赖它们。

> **假设 (R) 残差近单位结构**：目标线性核 $W$ 出现在残差块 `x = x + W·core(x)` 中，并以 $W\approx I$ 初始化。
> **假设 (NU) 近酉性**（核心）：保留子空间块 $A:=Q^\dagger W Q$ 的奇异值满足 $\sigma_j(A)=1+O(\varepsilon)$，$\varepsilon\ll 1$。
> **假设 (N) 近正规性**：$\mathrm{col}(W)\approx\mathrm{row}(W)$（由 (R) 保证），使单边与双边子空间投影一致。

**为什么是"近酉"而不是"近单位阵"**：量子端只能执行酉演化 $e^{-iH}$。把一个矩阵 $A$ 替换成最近的酉矩阵，丢掉的恰好是它的"拉伸"成分，其大小**精确**等于奇异值偏离 1 的程度（定理 4.1）。$W\approx I$ 是 (NU) 的一个特例（$I$ 是酉的）；但一旦训练让 $\sigma_j(A)$ 离开 1，量子化就开始丢一阶信息。**(NU) 是命根子，(R) 只是保证 (NU) 在初始化/零样本时成立的手段。**

> ⚠️ **重要推论（实验设计须知）**：对**实数** $W$，$A$ 的扰动 $A-I$ 一般同时含对称部分（拉伸→丢失）和反对称部分（旋转→保留），二者**都是一阶**。因此"零样本无损"只在 $W$ 接近**正交矩阵**时成立（$I$ 即其一例）；训练会放大对称拉伸，破坏 (NU)。

---

## 1. 记号

- 复 Stiefel 流形：$\mathrm{St}(k,n;\mathbb{C})=\{Q\in\mathbb{C}^{n\times k}\mid Q^\dagger Q=I_k\}$。
- 厄米矩阵空间记 $i\,\mathfrak{u}(k):=\{H\in\mathbb{C}^{k\times k}\mid H=H^\dagger\}$。
  （$U(k)$ 的李代数 $\mathfrak{u}(k)$ 是**反厄米**矩阵；$H$ 厄米 $\Leftrightarrow -iH\in\mathfrak{u}(k)\Leftrightarrow e^{-iH}\in U(k)$。）
- 子空间投影 $P_Q:=QQ^\dagger$（幂等，秩 $k$），双边投影 $\Pi_Q(W):=QQ^\dagger W QQ^\dagger=Q(Q^\dagger WQ)Q^\dagger$。
- $\|A\|_F=\sqrt{\mathrm{Tr}(A^\dagger A)}$；$\sigma_j(\cdot)$ 表奇异值。

↩ *替换原 §0、定义 0.2 中"$\mathfrak{u}(k)=$ 厄米集合"的记号错误。*

---

## 2. 混合算子与规范不变性（原命题正确，保留）

**定义 2.1（混合算子）** 对 $Q\in\mathrm{St}(k,n;\mathbb{C})$、$H=H^\dagger$：
$$\mathcal{O}(Q,H):=Q\,e^{-iH}\,Q^\dagger\in\mathrm{End}(\mathbb{C}^n).$$

**命题 2.2（子空间酉性 + 投影表示）**
(i) $\mathcal{O}$ 限制在 $\mathrm{span}(Q)$ 上是酉等距；在 $\mathrm{span}(Q)^\perp$ 上为零，故作为 $\mathbb{C}^n$ 上的算子是**部分等距**，秩 $k$。
(ii) $\mathcal{O}(Q,H)=P_Q\,e^{-i\tilde H}\,P_Q$，其中 $\tilde H=QHQ^\dagger$。
*证.* (i) 对 $v=Qx$，$\mathcal{O}v=Qe^{-iH}x$，$\|\mathcal{O}v\|=\|x\|=\|v\|$。(ii) 用 $\tilde H^m=QH^mQ^\dagger$（因 $Q^\dagger Q=I_k$）逐项展开 $e^{-i\tilde H}=I_n+Q(e^{-iH}-I_k)Q^\dagger$，两侧夹 $P_Q$ 并用 $P_QQ=Q$ 即得。∎

**定理 2.3（规范不变性 / gauge invariance）** $\forall R\in U(k)$：
$$\mathcal{O}(QR,\,R^\dagger HR)=\mathcal{O}(Q,H).$$
*证.* $e^{-iR^\dagger HR}=R^\dagger e^{-iH}R$，代入消去 $RR^\dagger=I_k$。∎

**注 2.4（关于"丛"的正确表述）** 由定理 2.3，$\mathcal{O}$ 在等价关系 $(Q,H)\sim(QR,R^\dagger HR)$ 下良定，即它是**伴随丛** $\mathrm{St}(k,n;\mathbb{C})\times_{U(k)} i\mathfrak{u}(k)$ 上的一个**规范不变映射**。
> 注意：它**不是**任何丛的"截面 (section)"。这里引入伴随丛**不提供额外定理**，仅是"规范不变性"的几何语言；写论文时按此表述，勿称 section。

↩ *替换原命题 1.1、定理 2.1（内容正确，仅修语言）与推论 2.1（删去"smooth section"范畴错误）。*

---

## 3. 转移映射 $\mathcal{T}:W\mapsto(Q,H)$（修正生成元定义）

**定义 3.1（转移映射）** 给定 $W$，定义 $\mathcal{T}(W)=(Q,H)$：
1. **子空间标架**：SVD $W=U\Sigma V^\dagger$，取 $Q=U_{:,:k}$。
2. **子空间块**：$A:=Q^\dagger W Q\in\mathbb{C}^{k\times k}$。
3. **最近酉投影**：极分解 $A=U_AP_A$，$U_A\in U(k)$，$P_A=(A^\dagger A)^{1/2}\succeq0$。
   （数值实现：$A=X\Sigma_A Y^\dagger\Rightarrow U_A=XY^\dagger$。）
4. **生成元（厄米）**：
$$\boxed{H:=i\log(U_A)}\qquad(\text{主对数}),\qquad e^{-iH}=U_A.$$
   小角度等价式（(NU) 下）：$H\approx \tfrac{i}{2}(A-A^\dagger)$（$i\times$**反厄米部分**）。

**假设 3.2（谱条件，保证主对数单值）** $\sigma(U_A)\cap(-\infty,0]=\varnothing$。由 (R)/(NU)（$U_A$ 特征值聚集于 $1$）自动满足。

> 🔧 **与代码的差异（必须按本定义实现）**：原代码 `H=(A+A†)/2`（厄米部分）是**错误**的，它在 $A\approx I$ 时给出 $H\approx I$、$e^{-iH}\approx e^{-i}I$（一个全局相位），并非旋转生成元。正确实现见 Step 3–4。

↩ *替换原定义 3.1（Step 4 与代码统一为 $i\log U_A$）。*

---

## 4. 精确误差分解（修正主定理 6.1）

这是全理论的核心定量结果。**关键修正：原 $C\|H\|_F^2$ 界一般情形为假**（反例 $A=2I$：左 $>0$、右 $=0$）。正确的界把量子化误差写成**精确的奇异值偏差**，不依赖任何近似。

**引理 4.1（最近酉的精确误差，Fan–Hoffman）** 对任意 $A$，
$$\min_{U\in U(k)}\|A-U\|_F=\|A-U_A\|_F=\|P_A-I\|_F=\Big(\textstyle\sum_{j\le k}(\sigma_j(A)-1)^2\Big)^{1/2}.$$
*证.* $\|A-U\|_F^2=\|A\|_F^2+k-2\,\mathrm{Re}\,\mathrm{Tr}(U^\dagger A)$。写 $A=X\Sigma_AY^\dagger$，令 $Z=Y^\dagger U^\dagger X\in U(k)$，则 $\mathrm{Re}\,\mathrm{Tr}(U^\dagger A)=\sum_j\sigma_j(A)\,\mathrm{Re}(Z_{jj})\le\sum_j\sigma_j(A)$，等号取 $Z=I\Leftrightarrow U=XY^\dagger=U_A$。代回得 $\sum_j\sigma_j^2+k-2\sum_j\sigma_j=\sum_j(\sigma_j-1)^2$。∎

**定理 4.2（量子化误差分解 — 精确版）** 设 $(Q,H)=\mathcal{T}(W)$。则
$$
\big\|W-\mathcal{O}(Q,H)\big\|_F\;\le\;\underbrace{\big\|W-\Pi_Q(W)\big\|_F}_{\text{(i) 子空间截断}}\;+\;\underbrace{\big\|P_A-I\big\|_F}_{\text{(ii) 非酉投影（精确）}},
$$
其中第 (ii) 项 $=\big(\sum_{j\le k}(\sigma_j(A)-1)^2\big)^{1/2}$ 为**等式**；在假设 (N) 下第 (i) 项 $\le\big(\sum_{j>k}\sigma_j(W)^2\big)^{1/2}$。
*证.* 三角不等式 $\|W-\mathcal{O}\|\le\|W-\Pi_Q(W)\|+\|\Pi_Q(W)-\mathcal{O}\|$。第二项 $=\|Q(A-U_A)Q^\dagger\|_F=\|A-U_A\|_F$（$Q$ 列正交，Frobenius 不变），由引理 4.1 等于 $\|P_A-I\|_F$。第一项在 (N)（col≈row）下退化为 Eckart–Young 截断 $\sqrt{\sum_{j>k}\sigma_j(W)^2}$。∎

**推论 4.3（何时量子化"几乎无损"）**
- **近酉区 (NU)**：$\sigma_j(A)=1+O(\varepsilon)\Rightarrow$ 第 (ii) 项 $=O(\varepsilon\sqrt{k})$。
- **纯旋转充分条件**：若 $A-I$ 的**厄米部分**为 $O(\|H\|^2)$（即 $A$ 的扰动近似纯反厄米/纯旋转），则 $\sigma_j(A)=1+O(\|H\|^2)$，第 (ii) 项退化为 $O(\|H\|^2)$。**这才是原文想要的二阶界——它需要"扰动纯旋转"这个额外假设，并非无条件成立。**

> **物理解读（替换原"相位隆起/信息完备"叙述）**：量子化误差由两件事决定——保留多少维（截断 (i)）、保留的块离酉多远（非酉 (ii)）。零样本可用，当且仅当 $W$ 近酉使 (ii) 小。训练会增大 $A$ 的对称拉伸 → (ii) 增大；这是实验 B 误差随训练上升的正确解释，与"相位"无关。

↩ *替换原定义 5.1、引理 5.1、定理 6.1 及 merge_theory 中"$P_A-I=\tfrac12H^2$"那段（该式仅在"扰动纯旋转"假设下成立，已并入推论 4.3）。同时把"最优性"由 Eckart–Young 更正为 Fan–Hoffman 最近酉（引理 4.1）。*

---

## 5. 流形合并（修正传输维度与非对易归因）

设两个近酉核 $W_A,W_B$（如对类别子集微调得到），系数 $\alpha+\beta=1$。给出两条**诚实**路线及其误差。

### 5.1 公共子空间（修正维度）

**定义 5.1（覆盖标架）** $Q_C:=\mathrm{orth}\big([\,Q_A\ \ Q_B\,]\big)\in\mathrm{St}(k_C,n;\mathbb{C})$，$k_C\le 2k$。
> 🔧 **修正**：必须取**张成两者的并**的标架（$k_C\le 2k$，物理上 $\le\lceil\log_2 2k\rceil$ 比特），而非 $W_C$ 的 top-$k$。只有当 $\mathrm{span}(Q_C)\supseteq\mathrm{span}(Q_A)\cup\mathrm{span}(Q_B)$ 时，下面的平行传输才**无损**。原代码取 top-$k$ 会丢掉落在 $Q_C$ 外的生成元能量，这是合并掉点的一阶主因。

### 5.2 平行传输（Lift–Project–Restrict，正确，保留）

**定义 5.2** $H_A':=Q_C^\dagger(Q_AH_AQ_A^\dagger)Q_C$，$H_B'$ 同理。
**引理 5.3** $H_A'=(H_A')^\dagger$（保厄米）；且 $Q_CH_A'Q_C^\dagger=P_C(Q_AH_AQ_A^\dagger)P_C$，$P_C=Q_CQ_C^\dagger$。
**引理 5.4（传输无损条件）** 若 $\mathrm{span}(Q_A)\subseteq\mathrm{span}(Q_C)$，则 $Q_CH_A'Q_C^\dagger=Q_AH_AQ_A^\dagger$（提升的全局哈密顿量不变）。
*证.* 此时 $P_CQ_A=Q_A$，代入引理 5.3 即得。∎

### 5.3 两条合并路线

**路线 (a) 算子空间合并（基准）**：$W_C=\alpha W_A+\beta W_B$（仍近酉），对 $W_C$ 重跑 $\mathcal{T}$，得 $(Q_C^{(a)},H_C^{(a)})$。误差直接由定理 4.2 控制。
**路线 (b) 生成元空间合并（几何方案）**：$H_C=\alpha H_A'+\beta H_B'$，$\mathcal{O}_C=Q_Ce^{-iH_C}Q_C^\dagger$。

### 5.4 合并误差界（修正非线性项；对易子并非首阶）

**引理 5.4（平均-指数二阶展开）** $X,Y$ 厄米，$\alpha=\beta=\tfrac12$：
$$e^{-i\frac{X+Y}{2}}-\tfrac12\big(e^{-iX}+e^{-iY}\big)=-\tfrac18(X-Y)^2+O(3).$$
（证：二阶泰勒后用 $(X+Y)^2-2(X^2+Y^2)=-(X-Y)^2$。详见 `THEORY.md` 引理 5.4。）

**定理 5.5（合并误差，路线 b，$\alpha=\beta=\tfrac12$、传输无损）**
$$
\big\|W_C-\mathcal{O}_C\big\|_F\le\underbrace{\big\|W_C-\Pi_{Q_C}(W_C)\big\|_F}_{\text{(i) 截断+覆盖损失}}+\underbrace{\tfrac12\big(\|P_{A_A'}-I\|_F+\|P_{A_B'}-I\|_F\big)}_{\text{(ii) 各自非酉}}+\underbrace{\tfrac18\|H_A'-H_B'\|_F^2}_{\text{(iii) 指数非线性}}+O(3).
$$
第 (i) 项在 $\mathrm{span}(Q_C)\supseteq\mathrm{span}(Q_A)\cup\mathrm{span}(Q_B)$ 时仅剩截断（覆盖损失为 0），否则含**一阶**覆盖损失。

> 🔧 **修正（对易子位置）**：原文用 $e^Xe^Y\neq e^{X+Y}$（BCH）解释掉点是 **non-sequitur**——本构造"先平均后取指数"，**没有 $e^Xe^Y$ 乘积**。由引理 5.4，合并的首阶非线性代价是 $-\tfrac18(H_A'-H_B')^2$，由两生成元**差的平方**控制（对称项），**不是对易子** $[H_A',H_B']$。对易子只在"顺序相乘" $e^{-iX}e^{-iY}=e^{-i(X+Y)-\frac12[X,Y]+\cdots}$ 里出现；对称平均下 $[X,Y]$ 要到**三阶**才进来。

**掉点归因（替换"流形曲率/和乐群"叙述）**：合并精度损失 = (i) 覆盖损失（若 $k_C<2k$，一阶，常为主因）+ (ii) 各自非酉（同定理 4.2）+ (iii) $\tfrac18\|H_A'-H_B'\|_F^2$（二阶，模型越不同越大）。把全部掉点浪漫化为"流形内蕴曲率/和乐群"会掩盖真正可控的 (i)。

↩ *替换原定理 3、定义 7.1、命题 7.1、引理 8.1、定理 8.1 及"BCH/和乐群"段落。*

---

## 6. 迁移映射作为"反贫瘠高原"初始化（指数 → 多项式）

> 🔧 **定位修正**：不把本节当作"BP 免疫定理"，而当作一个**解析热启动初始化策略**。它的卖点是：把可训练区间的梯度尺度从"全局 $N$ 比特的指数级 $2^{-N}$"换成"固定 $\log_2 k$ 比特子空间的多项式级 $\mathrm{poly}(1/k)$"。这与恒等块初始化（Grant et al., 2019）同源，**新意在于近恒等初始点由经典预训练权重解析给出**，而非随机/手工构造。

### 6.1 两个不矛盾的事实（化解原定理 9 的自相矛盾）

原定理 9 把"2-design 的 $\mathrm{Var}\sim1/k^2$"与"实际集中在单位元邻域"当成互相推翻——其实它们是**地板**与**实际站位**，方向一致：

- **(F1) 最坏情形地板**：即便子空间内的演化被完全打乱成 $U(k)$ 上的酉 2-design，梯度方差也只是 $\mathrm{Var}\sim1/k^2$。因为**活动寄存器只有 $m=\log_2 k$ 个比特**，这是 $k$ 的多项式（而非 $N$ 的指数）。这是下界，不是退化。
- **(F2) 实际初始点更好**：迁移映射给出 $H_0=i\log U_A$，由 (NU) 有 $U_A\approx I_k$ 故 $\|H_0\|_F=O(\varepsilon)$。初始点落在 $U(k)$ 的**单位元邻域**——这是 2-design 的反面（最不打乱），梯度方差**不低于** (F1) 的地板。

结论：无论"最坏"还是"实际"，方差尺度都由**固定的 $k$** 决定，与全局比特数 $N$ 无关。

### 6.2 命题（初始化时刻的多项式梯度，可证）

**命题 6.1（子空间热启动的梯度下界）** 设训练自由度为子空间内参数化 $U(\boldsymbol\theta)=e^{-i(H_0+\sum_j\theta_j V_j)}\in U(k)$，$V_j\in i\mathfrak{u}(k)$，损失 $\mathcal{L}(\boldsymbol\theta)=\langle x|U^\dagger\tilde O U|x\rangle$，$\tilde O=Q^\dagger O Q$。则：
1. **维度约化**：全局 $N$ 比特的期望值精确约化到 $m=\log_2 k$ 比特寄存器上（命题 2.2(i)：$U(\boldsymbol\theta)|\Psi_0\rangle=Q\,e^{-i(\cdots)}|x\rangle$），故方差只对 $U(k)$（而非 $U(2^N)$）积分。
2. **最坏情形地板（2-design）**：$\mathrm{Var}[\partial_j\mathcal{L}]=\dfrac{\mathrm{Tr}([V_j,\tilde O]^2)}{k^2-1}=\Omega(1/k^2)$，多项式于 $k$，**独立于 $N$**。
3. **实际初始点（单位元邻域）**：在 $\|H_0\|=O(\varepsilon)$ 处，$U(\boldsymbol\theta)\approx I$，$\partial_j\mathcal{L}\big|_0\approx i\langle x|[V_j,\tilde O]|x\rangle=O(1)$（不随 $N$ 衰减）。
*证.* (1) 即命题 2.2(i) 的子空间约化；(2) 标准 $U(k)$ 上 Weingarten 二阶矩，分母为 $k^2-1$ 而非 $2^{2N}-1$；(3) 一阶展开，单位元邻域非 2-design，梯度为 $O(1)$ 量级。∎

> **"指数 → 多项式"的精确含义**：一个探索全 $2^N$ 维希尔伯特空间的通用 ansatz，梯度方差 $\sim2^{-N}$（**指数小于比特数 $N$**）；本初始化把训练锁进 SVD 选定的 $k$ 维子空间（$m=\log_2 k$ 比特，$k$ 由经典 SVD 固定、不随 $N$ 增长），方差 $\Omega(\mathrm{poly}(1/k))$（**多项式小于子空间维数 $k$，且与 $N$ 无关**）。作为初始化，它把可训练性从指数级搬到多项式级。

### 6.3 诚实边界（不要越界宣称）

- 命题 6.1 是**初始化时刻**的陈述。它保证你从一个梯度非平的好点出发；**不**等于保证整条优化轨迹都不进高原（轨迹是否一直留在良性区是经验问题）。
- 当前代码 $U$ 为冻结 buffer、**无 $\theta_j$**，属零样本迁移；命题 6.1 描述的是"若在此初始化上开放 $V_j$ 微调"的情形，是**少量学习**版本的理论支撑，列为下一步实现。
- 文献定位：等价于"经典热启动 + 恒等块初始化"，规避了随机/2-design 初始化的 $2^{-N}$ 高原；可对标 Grant et al. (2019)、Cerezo et al. (2021) 的局部/浅层论证。

↩ *替换原定理 9：由"免疫定理"改为"热启动初始化命题"，并化解 2-design 与单位元邻域的伪矛盾。*

---

## 7. 假设汇总（一页速查）

| 记号 | 内容 | 由谁保证 | 失效后果 |
|---|---|---|---|
| (R) | $W$ 在残差块中，$W\approx I$ 初始化 | 网络结构 + eye 初始化 | 离开近酉区 |
| (NU) | $\sigma_j(A)\approx1$（近酉，**核心**） | (R) 在初始化/零样本时 | 定理 4.2 第 (ii) 项变一阶主项 |
| (N) | $\mathrm{col}(W)\approx\mathrm{row}(W)$ | (R) | 单边/双边投影不一致，(i) 项失真 |
| 3.2 | $\sigma(U_A)\cap(-\infty,0]=\varnothing$ | (NU)（$U_A\approx I$） | 主对数多值，$H$ 不光滑 |
| 5.1 | $\mathrm{span}(Q_C)\supseteq\mathrm{span}(Q_A\cup Q_B)$ | 取 $\mathrm{orth}([Q_A,Q_B])$ | 传输一阶覆盖损失 |

**训练会破坏 (NU)**：这是最需要警惕的——零样本时 (NU) 成立，但只要更新 $W$ 使奇异值离开 1，定理 4.2 的 (ii) 项就从二阶退化为一阶。实验 B 即此现象。

---

## 8. 与原 notebook 的对应（替换表）

| 原命题 | 状态 | 本文对应 |
|---|---|---|
| 命题 1.1 / 定理 2.1 | ✅ 正确，保留 | 命题 2.2 / 定理 2.3 |
| 推论 2.1（丛截面） | 🔧 语言修正 | 注 2.4（规范不变映射，非截面） |
| 定义 0.2（$\mathfrak{u}(k)$=厄米） | 🔧 记号修正 | §1（厄米=$i\mathfrak{u}(k)$） |
| 定义 3.1 Step 4（$H=i\log U_A$） | ✅ 正确，但代码未实现 | 定义 3.1（统一为 $i\log U_A$） |
| 引理 5.1（Eckart–Young 最优） | 🔧 改为最近酉 | 引理 4.1（Fan–Hoffman） |
| 定理 6.1（$C\|H\|^2$ 界） | 🔴 一般为假 | 定理 4.2（精确奇异值界）+ 推论 4.3 |
| 定理 3 / 定理 8.1（合并） | 🔧 维度+归因修正 | §5（覆盖标架 + 三项误差） |
| "BCH/和乐群"解释掉点 | 🔴 non-sequitur | 引理 5.4（首阶是 $\tfrac18\|H_A'-H_B'\|^2$，非对易子） |
| 定理 9（BP 免疫） | 🔧 重定位为初始化策略 | §6（命题 6.1：热启动初始化，指数→多项式） |
| "等距同构/度量守恒" | 🔴 过度宣称 | §0（近似有损嵌入） |

---

### 一句话总结（修正后的论文主命题）
> **残差块中近酉的线性核 $W$，可被其保留子空间上的最近酉 $e^{-iH}$（$H=i\log U_A$）替换，量子化误差精确等于子空间块奇异值偏离 1 的均方根（定理 4.2）；当 $W$ 近酉时该误差为高阶小量，故可零训练迁移到 $\log_2 k$ 比特电路。多个此类核可在覆盖标架 $Q_C=\mathrm{orth}([Q_A,Q_B])$ 上平行传输后于生成元空间线性合并，掉点由"覆盖损失 + 非酉 + 生成元差的平方 $\tfrac18\|H_A'-H_B'\|^2$"三者解析刻画。**
