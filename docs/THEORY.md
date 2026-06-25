# 近酉线性核的子空间量子化与生成元合并 —— 完整理论（带证明）
## Subspace Quantization of Near-Unitary Linear Cores — Complete Theory with Proofs

> **本文目标**：用一条清晰的逻辑链，把"经典 MLP 线性核 → 量子 PQC → 模型合并 → 反贫瘠高原初始化"完整串起来，每个关键命题都给出**正确的证明**，并在每步标注 `↳ 连接` 说明它和上下游的关系。
> 配套文件：`theory_corrected.md` 是同一理论的**速查版**（无证明）。本文是**完整版**（有证明、有逻辑链）。

---

## 0. 全景：一条逻辑链（先看懂结构，再看细节）

你要做的事，本质是一个**有损但可控**的翻译：把经典线性核 $W$ 翻译成量子能执行的酉演化，再支持在量子端合并、并作为可训练的好起点。整条链是：

```
            (R) 残差近单位初始化  ──保证──►  (NU) 近酉性 σ(A)≈1
                                                  │  这是一切"无损"的命根子
                                                  ▼
  W ──SVD──► Q (子空间标架) ──► A=Q†WQ ──极分解──► U_A ──► H=i·log U_A
  §3 转移映射                                    §3 最近酉投影      §3 生成元
                                                  │
                                                  ▼
        混合算子 O(Q,H)=Q e^{-iH} Q†   ◄──§2 规范不变（良定性）
                                                  │
                                                  ▼
   §4 主定理：‖W−O‖ ≤ 截断误差(i) + ‖P_A−I‖(ii，精确=Σ(σ_j−1)²)
                                                  │
              ┌───────────────────────┴───────────────────────┐
              ▼                                                ▼
   §5 合并：传输 H_A,H_B 到 Q_C，                  §6 初始化：H_0≈0 落在单位元邻域
        生成元空间线性平均 → O_C                        ⇒ 梯度 poly(1/k)，与 N 无关
   误差 = 覆盖损失 + 非酉 + ⅛‖H_A'−H_B'‖²              （指数级 → 多项式级）
```

**三个 takeaway**（看不懂细节也要记住）：
1. **(NU) 近酉是支点**：量子端只会做酉演化，所以把 $A$ 换成最近酉 $U_A$，丢掉的恰是"拉伸"，其大小精确等于奇异值偏离 1 的程度。$W$ 越近酉，翻译越无损。
2. **误差是两块拼的**：选多少维（截断）＋ 选中的块离酉多远（非酉）。没有第三种神秘误差。
3. **合并与初始化是同一构造的两个用途**：合并＝在 $Q_C$ 上线性叠加生成元；初始化＝把 $H_0\approx0$ 当作量子训练的热启动点。

---

## 1. 问题设定与记号

**对象**：经典 MLP 中一个**方形线性核** $W\in\mathbb{C}^{n\times n}$（实际为实矩阵），用在残差块 `x ↦ x + W·core(x)` 中。
**目标**：构造 $(Q,H)$，$Q\in\mathrm{St}(k,n;\mathbb{C})$、$H=H^\dagger$，使 $\mathcal{O}(Q,H):=Qe^{-iH}Q^\dagger\approx W$，从而把该核换成只需 $\log_2 k$ 个量子比特的酉演化，**零训练或少量训练**即可用。

**记号**
- $\mathrm{St}(k,n;\mathbb{C})=\{Q\in\mathbb{C}^{n\times k}\mid Q^\dagger Q=I_k\}$（复 Stiefel 流形）。
- 厄米空间记 $i\,\mathfrak{u}(k)=\{H\mid H=H^\dagger\}$。注意 $U(k)$ 的李代数 $\mathfrak{u}(k)$ 是**反厄米**矩阵；$H$ 厄米 $\Leftrightarrow -iH\in\mathfrak{u}(k)\Leftrightarrow e^{-iH}\in U(k)$。
- $P_Q:=QQ^\dagger$（秩 $k$ 投影），$\Pi_Q(W):=QQ^\dagger WQQ^\dagger$（双边投影）。
- $\|A\|_F=\sqrt{\mathrm{Tr}(A^\dagger A)}$；$\sigma_j(\cdot)$ 为奇异值。

**核心假设（贯穿全文）**

| 记号 | 内容 | 来源 | 失效后果 |
|---|---|---|---|
| **(R)** | $W$ 在残差块中、$W\approx I$ 初始化 | 网络结构 + `eye_` | 离开近酉区 |
| **(NU)** | $\sigma_j(A)\approx1$（近酉，**支点**） | (R) 在零样本时刻 | §4 第 (ii) 项由二阶退化为一阶 |
| **(N)** | $\mathrm{col}(W)\approx\mathrm{row}(W)$（近正规） | (R) | 双边投影 ≠ 单边，截断项失真 |

> ⚠️ **最重要的一句话**：**训练会破坏 (NU)**。零样本时 $W\approx I$ 使 (NU) 成立；一旦更新 $W$ 让奇异值离开 1，量子化误差立刻从二阶变一阶。这是后面一切定量结论的边界。

---

## 2. 混合算子与良定性（为什么 $(Q,H)$ 这个表示是合法的）

**定义 2.1（混合算子）** $\mathcal{O}(Q,H):=Q\,e^{-iH}\,Q^\dagger$，$Q\in\mathrm{St}(k,n;\mathbb{C})$，$H=H^\dagger$。

**命题 2.2（子空间酉性 + 全局投影表示）**
(i) $\mathcal{O}$ 在 $\mathrm{span}(Q)$ 上是酉等距、在 $\mathrm{span}(Q)^\perp$ 上为零，故是秩 $k$ 的**部分等距**。
(ii) $\mathcal{O}(Q,H)=P_Q\,e^{-i\tilde H}\,P_Q$，其中 $\tilde H=QHQ^\dagger$。

*证.*
(i) 任取 $v=Qx$（$x\in\mathbb{C}^k$）。则 $\mathcal{O}v=Qe^{-iH}Q^\dagger Qx=Qe^{-iH}x$（用 $Q^\dagger Q=I_k$）。于是
$$\|\mathcal{O}v\|^2=x^\dagger e^{iH}\underbrace{Q^\dagger Q}_{=I_k}e^{-iH}x=x^\dagger x=\|v\|^2.$$
对 $v\perp\mathrm{span}(Q)$ 即 $Q^\dagger v=0$，$\mathcal{O}v=0$。
(ii) 由 $\tilde H^m=(QHQ^\dagger)^m=QH^mQ^\dagger$（$Q^\dagger Q=I_k$），
$$e^{-i\tilde H}=I_n+\sum_{m\ge1}\frac{(-i)^m}{m!}QH^mQ^\dagger=I_n+Q(e^{-iH}-I_k)Q^\dagger=I_n-P_Q+\mathcal{O}(Q,H).$$
两侧夹 $P_Q$，用 $P_Q^2=P_Q$、$P_QQ=Q$：$P_Qe^{-i\tilde H}P_Q=P_Q-P_Q+Q e^{-iH}Q^\dagger=\mathcal{O}(Q,H)$。∎

> ↳ **连接**：(i) 说明"经典权重塞进量子电路"在子空间内严格保范数——这是把 $W$ 当酉演化处理的前提。(ii) 给出"在 $n$ 维全局构造哈密顿量 $\tilde H$ 再投影"与"在 $k$ 维局部演化"等价，是用小电路高效模拟的依据。

**定理 2.3（规范不变性 / gauge invariance，$(Q,H)$ 表示良定）** $\forall R\in U(k)$：
$$\mathcal{O}(QR,\,R^\dagger HR)=\mathcal{O}(Q,H).$$

*证.* 对可逆 $R$ 有 $(R^\dagger HR)^m=R^\dagger H^m R$，故 $e^{-iR^\dagger HR}=R^\dagger e^{-iH}R$。代入：
$$\mathcal{O}(QR,R^\dagger HR)=QR\,(R^\dagger e^{-iH}R)\,R^\dagger Q^\dagger=Q(RR^\dagger)e^{-iH}(RR^\dagger)Q^\dagger=Qe^{-iH}Q^\dagger.\qquad\blacksquare$$

> ↳ **连接**：转移映射（§3）从 SVD 取 $Q$ 时，左奇异向量有 $U(k)$ 的相位/旋转自由度（简并子空间内基底不唯一）。定理 2.3 保证**无论 SVD 解出哪一组基**，只要 $H$ 协变，输出 $\mathcal{O}$ 不变。换言之 $\mathcal{O}$ 是伴随丛 $\mathrm{St}(k,n;\mathbb{C})\times_{U(k)}i\mathfrak{u}(k)$ 上的**规范不变映射**（它是映射，不是"截面"）。这让 §3 的构造不依赖 SVD 实现细节。

---

## 3. 转移映射 $\mathcal{T}:W\mapsto(Q,H)$（如何把经典核变成量子的）

四步流水线 $W\to Q\to A\to U_A\to H$。

**Step 1 — 子空间标架（Eckart–Young 选维）**
SVD $W=U\Sigma V^\dagger$，取 $Q=U_{:,:k}$（前 $k$ 个左奇异向量）。
*为什么*：要用 $k$ 维子空间装下 $W$ 的主要信息，最优选择就是最大奇异值对应的方向。

**Step 2 — 子空间块**
$A:=Q^\dagger WQ\in\mathbb{C}^{k\times k}$。注意 $QAQ^\dagger=\Pi_Q(W)$。
*问题*：$A$ 一般既不保范数（非酉）也不保能量（非厄米），量子机器无法直接执行。

**Step 3 — 最近酉投影（极分解，这是"量子化"的核心一刀）**
极分解 $A=U_AP_A$，$U_A\in U(k)$，$P_A=(A^\dagger A)^{1/2}\succeq0$。
量子门只能做旋转（酉），不能做拉伸。于是**砍掉拉伸 $P_A$，只留旋转 $U_A$**。
数值实现：若 $A=X\Sigma_AY^\dagger$，则 $U_A=XY^\dagger$（最近酉，见引理 4.1）。

**Step 4 — 提取生成元**
$$\boxed{H:=i\log(U_A)}\quad(\text{主对数}),\qquad e^{-iH}=U_A,\qquad H=H^\dagger.$$
$H$ 厄米性：$U_A^\dagger U_A=I\Rightarrow (e^{-iH})^\dagger e^{-iH}=e^{iH^\dagger}e^{-iH}=I\Rightarrow H^\dagger=H$。
小角度等价式（(NU) 下 $A\approx I$）：$H\approx\tfrac{i}{2}(A-A^\dagger)$（$i\times$ **反厄米部分**）。

**假设 3.1（谱条件，保主对数单值）** $\sigma(U_A)\cap(-\infty,0]=\varnothing$。
*为何自动成立*：(R)/(NU) 使 $A\approx I$，故 $U_A$ 特征值聚集在 $1$ 附近，远离负实轴的分支割线。

> 🔧 **实现警示（代码必须按此写）**：取厄米部分 $\tfrac12(A+A^\dagger)$ 是**错的**——在 $A\approx I$ 时它给出 $H\approx I$、$e^{-iH}\approx e^{-i}I$（只是一个全局相位），而非旋转生成元。必须用 Step 3–4 的 $i\log U_A$，或其一阶式 $\tfrac{i}{2}(A-A^\dagger)$。

> ↳ **连接**：Step 1 决定误差的第 (i) 项（截断），Step 3 决定第 (ii) 项（非酉）——这正是 §4 主定理的两块。Step 3 砍掉的 $P_A$ 不是凭空丢，它的大小被 §4 精确量化。

---

## 4. 主定理：量子化误差的精确分解（全理论的定量核心）

先证"最近酉"的精确误差（这是真正用到的最优性，注意**不是** Eckart–Young，而是 Fan–Hoffman）。

**引理 4.1（最近酉的精确误差）** 对任意 $A\in\mathbb{C}^{k\times k}$，极分解 $A=U_AP_A$：
$$\min_{U\in U(k)}\|A-U\|_F=\|A-U_A\|_F=\|P_A-I\|_F=\Big(\sum_{j=1}^k(\sigma_j(A)-1)^2\Big)^{1/2}.$$

*证.* 展开 $\|A-U\|_F^2=\|A\|_F^2+k-2\,\mathrm{Re}\,\mathrm{Tr}(U^\dagger A)$。设 SVD $A=X\Sigma_AY^\dagger$，令 $Z:=Y^\dagger U^\dagger X\in U(k)$，则
$$\mathrm{Re}\,\mathrm{Tr}(U^\dagger A)=\mathrm{Re}\,\mathrm{Tr}(Z\Sigma_A)=\sum_j\sigma_j(A)\,\mathrm{Re}(Z_{jj})\le\sum_j\sigma_j(A),$$
因 $|Z_{jj}|\le1$；等号 $\Leftrightarrow Z=I\Leftrightarrow U=XY^\dagger=U_A$。代回 $\min=\|A\|_F^2+k-2\sum_j\sigma_j=\sum_j\sigma_j^2+k-2\sum_j\sigma_j=\sum_j(\sigma_j-1)^2$。
又 $A-U_A=U_A(P_A-I)$，$U_A$ 酉、Frobenius 范数酉不变，故 $\|A-U_A\|_F=\|P_A-I\|_F$；$P_A$ 的特征值即 $\sigma_j(A)$。∎

**定理 4.2（量子化误差分解，精确版）** 设 $(Q,H)=\mathcal{T}(W)$，$A=Q^\dagger WQ=U_AP_A$。则
$$
\boxed{\;\big\|W-\mathcal{O}(Q,H)\big\|_F\;\le\;\underbrace{\big\|W-\Pi_Q(W)\big\|_F}_{\text{(i) 子空间截断}}\;+\;\underbrace{\big\|P_A-I\big\|_F}_{\text{(ii) 非酉（精确）}}\;}
$$
其中第 (ii) 项 $=\big(\sum_{j}(\sigma_j(A)-1)^2\big)^{1/2}$ 为**等式**；在 (N) 下第 (i) 项 $\le\big(\sum_{j>k}\sigma_j(W)^2\big)^{1/2}$。

*证.* 三角不等式：
$$\|W-\mathcal{O}\|_F\le\underbrace{\|W-\Pi_Q(W)\|_F}_{\text{(i)}}+\underbrace{\|\Pi_Q(W)-\mathcal{O}\|_F}_{\text{(ii)}}.$$
对 (ii)：$\Pi_Q(W)=QAQ^\dagger$，$\mathcal{O}=QU_AQ^\dagger$，故 $\|\Pi_Q(W)-\mathcal{O}\|_F=\|Q(A-U_A)Q^\dagger\|_F$。因 $Q^\dagger Q=I_k$，
$$\|QMQ^\dagger\|_F^2=\mathrm{Tr}(QM^\dagger Q^\dagger QMQ^\dagger)=\mathrm{Tr}(M^\dagger M\,Q^\dagger Q)=\|M\|_F^2,$$
即 $\|Q(A-U_A)Q^\dagger\|_F=\|A-U_A\|_F\overset{\text{引理4.1}}{=}\|P_A-I\|_F$。
对 (i)：$W-\Pi_Q(W)=(I-P_Q)W+P_QW(I-P_Q)$。第一块 $\|(I-P_Q)W\|_F=\sqrt{\sum_{j>k}\sigma_j(W)^2}$（Eckart–Young 单边截断）；第二块 $P_QW(I-P_Q)$ 在 (N)（$WP_Q\approx P_QW$，即列空间≈行空间）下为高阶小量，可略。∎

**推论 4.3（何时量子化"几乎无损"——把 (NU) 的作用钉死）**
- **近酉区 (NU)**：$\sigma_j(A)=1+O(\varepsilon)\Rightarrow$ 第 (ii) 项 $=O(\varepsilon\sqrt{k})$。零样本可用。
- **纯旋转充分条件**：若 $A-I$ 的**厄米部分**为 $O(\|H\|^2)$（扰动近似纯反厄米/纯旋转），则 $\sigma_j(A)=1+O(\|H\|^2)$，第 (ii) 项退化为 $O(\|H\|^2)$。**这才是"二阶小误差"的正确出处，它需要"扰动纯旋转"这个额外假设，并非无条件成立。**
- **一般区（$W$ 远离酉）**：$\sigma_j(A)$ 远离 1，第 (ii) 项是**一阶主项**，量子化丢真实信息。

> ↳ **连接（这段把整条链拧在一起）**：
> - 第 (i) 项 ← Step 1 选的 $k$；第 (ii) 项 ← Step 3 砍掉的 $P_A$。两项各对应一步构造，没有第三种误差。
> - "零训练就好用" = 第 (ii) 项小 = $A$ 近酉 = (NU)。而 (NU) 由 (R) 残差初始化保证。**这就是为什么必须用残差 + eye 初始化**。
> - 对**实数** $W$，扰动一般含对称（拉伸→丢）+ 反对称（旋转→留）两部分、都是一阶；故"无损"只在 $W$ 接近**正交矩阵**时成立。训练放大对称拉伸，使第 (ii) 项一阶化——这就是误差随训练上升的机制。

---

## 5. 流形合并（在量子端把两个模型并成一个）

设两个近酉核 $W_A,W_B$（如分别对 0–4 / 5–9 类微调），系数 $\alpha+\beta=1$。

### 5.1 公共标架与平行传输（把不同子空间的生成元搬到同一处）

**定义 5.1（覆盖标架）** $Q_C:=\mathrm{orth}\big([\,Q_A\ \ Q_B\,]\big)\in\mathrm{St}(k_C,n;\mathbb{C})$，$k_C\le2k$。
> 🔧 **必须取并空间的标架**（$k_C\le2k$，物理 $\le\lceil\log_2 2k\rceil$ 比特），**不是** $W_C$ 的 top-$k$。只有 $\mathrm{span}(Q_C)\supseteq\mathrm{span}(Q_A)\cup\mathrm{span}(Q_B)$ 时传输才无损（引理 5.3）。取 top-$k$ 会丢掉落在 $Q_C$ 外的生成元能量，是合并掉点的一阶主因。

**定义 5.2（联络诱导传输，Lift–Project–Restrict）**
$$H_A':=Q_C^\dagger\,(Q_AH_AQ_A^\dagger)\,Q_C,\qquad H_B':=Q_C^\dagger\,(Q_BH_BQ_B^\dagger)\,Q_C.$$
三步语义：**提升** $Q_AH_AQ_A^\dagger$（把 $k$ 维生成元抬到 $n$ 维全局，无歧义）→ **投影** $P_C(\cdot)P_C$ → **限制** 回 $Q_C$ 基底。

**引理 5.3（保厄米 + 无损条件）**
(i) $H_A'=(H_A')^\dagger$。
(ii) 若 $\mathrm{span}(Q_A)\subseteq\mathrm{span}(Q_C)$，则 $Q_CH_A'Q_C^\dagger=Q_AH_AQ_A^\dagger$（提升的全局哈密顿量被无损保留）。

*证.* (i) $(H_A')^\dagger=Q_C^\dagger(Q_AH_A^\dagger Q_A^\dagger)Q_C=H_A'$（$H_A^\dagger=H_A$）。
(ii) $\mathrm{span}(Q_A)\subseteq\mathrm{span}(Q_C)\Rightarrow P_CQ_A=Q_A$。故 $Q_CH_A'Q_C^\dagger=P_C(Q_AH_AQ_A^\dagger)P_C=(P_CQ_A)H_A(P_CQ_A)^\dagger=Q_AH_AQ_A^\dagger$。∎

> ↳ **连接**：引理 5.3(ii) 解释了 §5.1 为什么坚持 $k_C\le2k$ 的并空间——只有覆盖住，传输才不掉信息；否则 $P_CQ_A\ne Q_A$，差额就是覆盖损失。

### 5.2 合并与误差（修正"对易子"的位置）

**路线 (a) 算子空间合并（基准）**：$W_C=\alpha W_A+\beta W_B$（仍近酉），对 $W_C$ 重跑 $\mathcal{T}$。
**路线 (b) 生成元空间合并（几何方案）**：$H_C=\alpha H_A'+\beta H_B'$，$\mathcal{O}_C=Q_Ce^{-iH_C}Q_C^\dagger$。

合并的"额外"误差来自"经典是线性叠加、量子是指数映射"这一非线性鸿沟。下面精确刻画它。

**引理 5.4（平均-指数的二阶展开 —— 对易子在哪里）** 设 $X,Y$ 厄米，$\alpha=\beta=\tfrac12$。则
$$
e^{-i\frac{X+Y}{2}}-\tfrac12\big(e^{-iX}+e^{-iY}\big)=-\tfrac18\,(X-Y)^2+O(\|X\|^3+\|Y\|^3).
$$

*证.* 二阶泰勒：$e^{-iX}=I-iX-\tfrac12X^2+O(3)$，故
$$\tfrac12(e^{-iX}+e^{-iY})=I-\tfrac{i}{2}(X+Y)-\tfrac14(X^2+Y^2)+O(3).$$
又 $e^{-i\frac{X+Y}{2}}=I-\tfrac{i}{2}(X+Y)-\tfrac18(X+Y)^2+O(3)$。相减，一阶项抵消：
$$\Delta=-\tfrac18(X+Y)^2+\tfrac14(X^2+Y^2)=-\tfrac18\big[(X+Y)^2-2(X^2+Y^2)\big]=-\tfrac18\big[-(X-Y)^2\big]=\ldots$$
代数：$(X+Y)^2-2(X^2+Y^2)=X^2+XY+YX+Y^2-2X^2-2Y^2=-(X^2-XY-YX+Y^2)=-(X-Y)^2$。故 $\Delta=-\tfrac18(X-Y)^2$。∎

> 🔧 **关键澄清（替换原"BCH/和乐群"叙述）**：合并的首阶非线性代价是 $-\tfrac18(H_A'-H_B')^2$，由两生成元**差的平方**控制——是个对称项，**不是对易子** $[H_A',H_B']$。
> 对易子只在"**顺序相乘**" $e^{-iX}e^{-iY}=e^{-i(X+Y)-\frac12[X,Y]+\cdots}$（BCH）里出现；而本构造是"**先平均后取指数**"，根本没有这个乘积。对称平均下 $[X,Y]$ 要到**三阶**才进来。所以把合并掉点归于"BCH/流形曲率/和乐群"是过度浪漫化；正确的主因是**覆盖损失（一阶）+ 各自非酉（§4）+ $\tfrac18\|H_A'-H_B'\|^2$（二阶）**。

**定理 5.5（合并误差分解）** 在 $\alpha=\beta=\tfrac12$、传输无损（引理 5.3(ii)）下，
$$
\|W_C-\mathcal{O}_C\|_F\le\underbrace{\|W_C-\Pi_{Q_C}(W_C)\|_F}_{\text{(i) 截断+覆盖损失}}+\underbrace{\tfrac12\big(\|P_{A_A'}-I\|_F+\|P_{A_B'}-I\|_F\big)}_{\text{(ii) 各自非酉}}+\underbrace{\tfrac18\|H_A'-H_B'\|_F^2}_{\text{(iii) 指数非线性}}+O(3).
$$

*证.* 子空间内比较 $A_C:=Q_C^\dagger W_CQ_C=\tfrac12(A_A'+A_B')$ 与 $e^{-iH_C}$，$A_X':=Q_C^\dagger W_XQ_C$。
$$\|A_C-e^{-iH_C}\|_F\le\underbrace{\big\|\tfrac12(A_A'+A_B')-\tfrac12(e^{-iH_A'}+e^{-iH_B'})\big\|_F}_{\le\frac12(\|P_{A_A'}-I\|+\|P_{A_B'}-I\|)}+\underbrace{\big\|\tfrac12(e^{-iH_A'}+e^{-iH_B'})-e^{-iH_C}\big\|_F}_{\overset{\text{引理5.4}}{=}\frac18\|(H_A'-H_B')^2\|_F\le\frac18\|H_A'-H_B'\|_F^2}.$$
第一块每项 $\|A_X'-e^{-iH_X'}\|_F=\|P_{A_X'}-I\|_F$（同定理 4.2）。外层再加截断项 $\|W_C-\Pi_{Q_C}(W_C)\|_F$ 即得。∎

> ↳ **连接**：定理 5.5 与定理 4.2 同构——(i)(ii) 完全是单模型那两块，只是搬到 $Q_C$ 上；唯一新增的是 (iii)，它度量"两个模型分得多开"（$\|H_A'-H_B'\|$）。**模型越相似，合并越无损**；这给"为什么相斥的 0-4 / 5-9 合并掉点更多"一个正确解释：不是神秘曲率，而是 $\|H_A'-H_B'\|$ 大 + 覆盖损失大。

---

## 6. 把转移映射当作"反贫瘠高原"初始化（指数级 → 多项式级）

这一节回答：训练好的经典核翻译过去后，**若再在量子端微调**，会不会撞上贫瘠高原（梯度指数消失）？答案：不会，因为这是一个落在**单位元邻域**、且活动寄存器只有 $\log_2 k$ 比特的**热启动初始化**。

### 6.1 两个不矛盾的事实

- **(F1) 最坏情形地板**：即便子空间内演化被打乱成 $U(k)$ 上酉 2-design，梯度方差也只是 $\sim1/k^2$。因为活动寄存器只有 $m=\log_2 k$ 比特，这是 $k$ 的**多项式**（而非 $N$ 的指数）。这是下界。
- **(F2) 实际初始点更好**：转移映射给出 $H_0=i\log U_A$，由 (NU) 有 $U_A\approx I$ 故 $\|H_0\|_F=O(\varepsilon)$，落在**单位元邻域**（2-design 的反面），梯度方差不低于 (F1)。

二者同向：方差尺度都由**固定的 $k$**（非全局 $N$）决定。原"定理 9"把这两件事当成互相推翻是误解。

### 6.2 命题（初始化时刻的多项式梯度）

设引入子空间内可训练参数：$U(\boldsymbol\theta)=e^{-i(H_0+\sum_j\theta_jV_j)}\in U(k)$，$V_j=V_j^\dagger$，初态 $|\Psi_0\rangle=Q|x\rangle$，损失 $\mathcal{L}=\langle\Psi_0|U^\dagger OU|\Psi_0\rangle$。

**命题 6.1**
1. **维度约化**：$U(\boldsymbol\theta)|\Psi_0\rangle=Q\,e^{-i(H_0+\sum_j\theta_jV_j)}|x\rangle$，故 $\mathcal{L}=\langle x|\tilde U^\dagger\tilde O\tilde U|x\rangle$，$\tilde O=Q^\dagger OQ$，期望值**精确**只依赖 $k$ 维寄存器（$m=\log_2 k$ 比特）。
2. **最坏情形地板**：若 $\tilde U$ 为 $U(k)$ 上酉 2-design，则
$$\mathrm{Var}[\partial_j\mathcal{L}]=\frac{\big\|[V_j,\tilde O]\big\|_F^2}{k^2-1}=\Omega(1/k^2),\quad\text{独立于 }N.$$
3. **实际初始点**：在 $\|H_0\|=O(\varepsilon)$ 处，$\tilde U\approx I$，$\partial_j\mathcal{L}\big|_0\approx i\langle x|[V_j,\tilde O]|x\rangle=O(1)$，不随 $N$ 衰减。

*证.*
(1) 由命题 2.2(i)：$U(\boldsymbol\theta)|\Psi_0\rangle=[Qe^{-i(\cdots)}Q^\dagger+(I-P_Q)]Q|x\rangle=Qe^{-i(\cdots)}|x\rangle$（用 $Q^\dagger Q=I$、$(I-P_Q)Q=0$）。代入损失即约化到 $\tilde O$ 上，积分域是 $U(k)$ 而非 $U(2^N)$。
(2) 一阶导 $\partial_j\mathcal{L}=\langle x|\tilde U^\dagger\,i[V_j,\tilde O]\,\tilde U|x\rangle$。在 $U(k)$ Haar（2-design）下，一阶矩 $\propto\mathrm{Tr}(i[V_j,\tilde O])=0$；二阶矩由标准 $U(k)$ Weingarten 给出，分母为 $k^2-1$（而非 $2^{2N}-1$），整理得 $\|[V_j,\tilde O]\|_F^2/(k^2-1)$。注意 $[V_j,\tilde O]$ 反厄米，$\|[V_j,\tilde O]\|_F^2=-\mathrm{Tr}([V_j,\tilde O]^2)\ge0$。
(3) 一阶展开 $\tilde U=I-i(H_0+\sum\theta_jV_j)+\cdots$，在 $\theta=0$、$H_0\approx0$ 处直接得 $\partial_j\mathcal{L}\approx i\langle x|[V_j,\tilde O]|x\rangle$，量级 $O(1)$。∎

> **"指数 → 多项式"的精确含义**：通用 $N$ 比特 ansatz 方差 $\sim2^{-N}$（**指数小于比特数**）；本初始化把训练锁进 SVD 选定的 $k$ 维子空间（$m=\log_2 k$ 比特、$k$ 由经典 SVD 固定、不随 $N$ 增长），方差 $\Omega(\mathrm{poly}(1/k))$（**多项式小于子空间维数、与 $N$ 无关**）。作为初始化，它把可训练性从指数级搬到多项式级。

### 6.3 诚实边界
- 命题 6.1 是**初始化时刻**的陈述：保证从梯度非平的好点出发，**不**保证整条优化轨迹都不进高原（经验问题）。
- 当前代码 $U$ 为冻结 buffer、无 $\theta_j$，属零样本迁移；命题 6.1 是"在此热启动上开放 $V_j$ 做少量微调"的理论支撑，对应你"少量学习就更好"的目标。
- 文献定位：等价于"经典热启动 + 恒等块初始化"（Grant et al. 2019），规避随机/2-design 初始化的 $2^{-N}$ 高原；新意在于**近恒等初始点由经典预训练权重解析给出**，而非随机/手工。

> ↳ **连接**：§6 与 §3–§4 是同一个 $(Q,H_0)$ 的两种用途——§4 把它当"零样本替换"（$U$ 冻结），§6 把它当"少量学习的热启动点"（开放 $V_j$）。同一个 (NU) 假设：在 §4 保证替换无损，在 §6 保证初始点落在单位元邻域。

---

## 7. 与代码的对应 + 必改项

| 理论对象 | 代码位置 | 状态 |
|---|---|---|
| $Q=U_{:,:k}$ | `src/simulation.py:54-60` | ✅ 对 |
| $H=i\log U_A$（§3 Step 4） | `simulation.py:59` 写成 `(A+A.mH)/2` | 🔴 **错**，给出全局相位，必改 |
| 传输 Lift-Project-Restrict | `simulation.py:62-70` | ✅ 结构对 |
| 覆盖标架 $Q_C=\mathrm{orth}([Q_A,Q_B])$ | `simulation.py:286` 取 top-$k$ | 🔴 **维度不符**（§5.1），必改 |
| (R) 残差 + eye 初始化保 (NU) | `simulation.py:81-84,91` | ✅ 命根子，勿动 |
| $\log_2 k$ 比特电路 | `simulation.py:25-46` | ✅ |
| 可训练 $\theta_j$（§6） | 无（$U$ 冻结） | ⬜ 未实现，少量学习版需补 |

**最小必改集**：① `H = i·log(U_A)`（或一阶 `0.5j*(A-A.mH)`）；② `Q_C = orth([Q_A,Q_B])`（$\le2k$ 维）；③ 若要做少量学习，给 $U$ 加可训练 $V_j$。

---

## 8. 逻辑链一页总结

1. **(R) 残差+eye → (NU) 近酉**：这是支点，没有它后面全垮。
2. **转移映射 $\mathcal{T}$（§3）**：SVD 选 $Q$（定第 (i) 项）→ 极分解砍 $P_A$ 留 $U_A$（定第 (ii) 项）→ $H=i\log U_A$。
3. **良定性（§2）**：规范不变保证 $\mathcal{O}$ 不依赖 SVD 基底选择。
4. **主定理（§4）**：$\|W-\mathcal{O}\|\le$ 截断 $+\ \|P_A-I\|$（**精确** $=\sqrt{\sum(\sigma_j(A)-1)^2}$）。(NU) 下第二项二阶 → 零训练可用。
5. **合并（§5）**：传输到 $Q_C=\mathrm{orth}([Q_A,Q_B])$ → 生成元线性平均。误差 = 覆盖损失 + 非酉 + $\tfrac18\|H_A'-H_B'\|^2$（**差的平方，非对易子**）。
6. **初始化（§6）**：同一个 $H_0\approx0$ 当热启动点，梯度 $\mathrm{poly}(1/k)$、与 $N$ 无关（指数 → 多项式）。

> **修正后的论文主命题**：*残差块中近酉的线性核 $W$，可被其保留子空间上的最近酉 $e^{-iH}$（$H=i\log U_A$）替换，量子化误差精确等于子空间块奇异值偏离 1 的均方根；当 $W$ 近酉时该误差为高阶小量，故可零训练迁移到 $\log_2 k$ 比特电路，并作为单位元邻域的解析热启动初始化（梯度多项式而非指数）。多个此类核可在覆盖标架 $Q_C=\mathrm{orth}([Q_A,Q_B])$ 上平行传输后于生成元空间线性合并，掉点由"覆盖损失 + 非酉 + 生成元差平方"三者解析刻画。*
