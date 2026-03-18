# 公式生成總表（Full Formula Sheet）

本文分成兩個理論模塊：

1. **WTRED / 隨機圖主線**
2. **Attention spectral probe 主線**

---

## A. WTRED / 隨機圖主線

### A1. 基本量

- 邊數：
\[
E_t
\]

- 三角形數：
\[
T_t
\]

- 楔形數：
\[
W_t = \sum_v \binom{d_t(v)}{2}
\]

- 全局三角密度 proxy：
\[
q_t = \frac{3T_t}{W_t}
\]

---

### A2. Clique-greedy 修正後的一階漂移框架

令當前殘餘圖近似為 \(G(n,r_t)\)。則：

\[
T_t \approx \binom{n}{3} r_t^3 \sim \frac{1}{6} r_t^3 n^3
\]

\[
W_t \approx n \binom{r_t n}{2} \sim \frac{1}{2} r_t^2 n^3
\]

故

\[
q_t = \frac{3T_t}{W_t} \approx r_t.
\]

---

### A3. 刪除一個大小 \(s\) 的 clique 時的主項

三角形變化：

\[
\Delta T_t
\approx
-\left[
\binom{s}{3} + \binom{s}{2}(n-s)r_t^2
\right]
\]

楔形變化：

\[
\Delta W_t
\approx
-s(s-1)\left(r_t n - \frac{s}{2}\right)
\]

一階漂移：

\[
q_{t+1} - q_t
\approx
\frac{r_t \Delta W_t - 3\Delta T_t}{W_t}
\]

符號判定關鍵：

\[
3\Delta T_t - r_t \Delta W_t
\approx
\frac{s(s-1)}{2}
\left(
 n r_t^2 + s(1+r_t-3r_t^2)-2
\right)
\]

當 \(s\ge 3\) 時，一階閉包下得到 \(q_t\) 下降。

---

### A4. Triangle adjacency graph 的 branching 參數

對 triangle adjacency graph \(\mathcal H_t\)：

\[
R_t \approx 2 n r_t^2
\]

因此 giant cluster 的臨界點由

\[
R_t \sim 1
\]

決定，得到

\[
r_c \sim n^{-1/2}.
\]

---

### A5. Decorrelation threshold

Local coupling 的誤差主項：

\[
\varepsilon_t = O\left(\frac{1}{n^2 r_t^3} + \frac{1}{n r_t}\right)
\]

當

\[
\frac{1}{n^2 r_t^3} \sim 1
\]

即

\[
r_t \sim n^{-2/3}
\]

mean-field / local coupling 失效。

---

### A6. Sharp threshold（iff）

局部弱收斂成立當且僅當：

\[
d_{\mathrm{TV}}(\mathcal N_k, \mathcal T_k^{\mathrm{GW}}) \to 0
\iff r_t \gg n^{-2/3}.
\]

---

### A7. Motif 準則

對局部 motif \(H\)，定義

\[
\Lambda_H(r,n) = n^{v(H)-2} r^{e(H)-1}.
\]

若某個 amplifying motif 滿足

\[
\Lambda_H(r,n) \not\to 0,
\]

則 local coupling 可能失效。

對 WTRED，最小 motif 為 diamond。

---

### A8. Fragmentation kernel（uniform labeled tree）

對大小 \(k\) 的均勻標記樹，隨機切一條邊，ordered split kernel：

\[
\Phi_k(i,k-i)
=
\frac{\binom{k}{i} i^{i-1}(k-i)^{k-i-1}}{(k-1)k^{k-2}}.
\]

Stirling 漸近：

\[
\Phi_k(i,k-i)
\asymp
C \cdot \frac{k^{3/2}}{i^{3/2}(k-i)^{3/2}}.
\]

---

## B. Attention spectral probe 主線

### B1. Attention 矩陣

對單 head attention：

\[
A \in \mathbb R^{T \times T}, \qquad \sum_j A_{ij} = 1.
\]

---

### B2. Row-centering

\[
X = A - \operatorname{rowmean}(A)
\]

用來去除 softmax 所帶來的 mean-field 背景偏置。

---

### B3. Correlation proxy

\[
C = \frac{1}{T} X X^\top
\]

以最大特徵值作為譜集中 proxy：

\[
\rho(C) = \lambda_{\max}(C).
\]

---

### B4. Entropy

單 head 平均 row entropy：

\[
S(A) = -\frac{1}{T} \sum_{i=1}^T \sum_{j=1}^T A_{ij}\log A_{ij}.
\]

---

### B5. Instability score

\[
\mathrm{Score} = \rho(C) \cdot \left(1 + \max(0, S_{\mathrm{floor}} - S(A))\right)
\]

當 \(\rho\) 上升且 entropy 下降時，分數會放大。

---

### B6. Temperature clamp

在外部最小實驗版本中，對 logits 做溫度調整：

\[
\tau = 1 + \alpha \max(0, \theta - S)
\]

或在譜版本：

\[
\tau = 1 + \alpha \max(0, \rho - \theta).
\]

採樣分佈：

\[
p = \operatorname{softmax}(z/\tau).
\]

---

### B7. 時間先行性

定義：

\[
t_\rho = \inf\{t : \rho_t > \theta\},
\qquad
 t_c = \inf\{t : \text{collapse occurs}\}
\]

lead-time：

\[
\Delta t = t_c - t_\rho.
\]

---

### B8. Functional criterion（弱形式）

令局部 correlation energy：

\[
\mathcal V = \sum_{e \sim e'} \mathrm{Cov}(\tau(e), \tau(e')).
\]

則 coupling failure 可由

\[
\frac{\mathcal V}{(n r^2)^2}
\]

的爆發來刻畫。

---

### B9. Spectral criterion（弱形式）

令

\[
(\mathcal K f)(e) = \sum_{e' \sim e} \mathrm{Cov}(\tau(e),\tau(e')) f(e').
\]

歸一化算子：

\[
\widetilde{\mathcal K} = \frac{\mathcal K}{(\mathbb E[\tau])^2}.
\]

弱形式判準：

\[
\rho(\widetilde{\mathcal K}) = o\left(\frac{1}{\log n}\right)
\Rightarrow d_{\mathrm{TV}} \to 0.
\]

---

## C. 重要分級

### 嚴格（Theorem-level）
- local coupling upper bound
- failure boundary lower bound
- sharp threshold iff（在 WTRED 主線中）

### 弱形式 / Proposition
- motif principle
- functional criterion
- spectral criterion (weak form)

### 猜想 / Conjecture
- sparse fragmentation PDE for WTRED triangle clusters
- universality transfer from conditioned GW trees to WTRED clusters
