"""
NEEP (Neural Estimator for Entropy Production) - Experiment 1: Bead-Spring Model

架構概覽：
  1. simulate_bead_spring()  - 用 Euler-Maruyama 模擬 Langevin 方程式，產生粒子軌跡
  2. BeadSpringDataset       - 把軌跡切成 (s_t, s_{t+1}) pairs，包成 PyTorch Dataset
  3. MLP                     - 3 個隱藏層（各 64 neurons），輸入 2N，輸出純量
  4. neep_loss()             - 計算 J(θ) = E[ΔS_θ - exp(-ΔS_θ)]，取負值做最小化
  5. train()                 - 訓練迴圈，20 epochs，Adam optimizer
  6. analytical_ep_rate()    - 用標準 OU 過程公式計算解析解 σ̇（已修正）
  7. plot_trajectories()     - 畫冷熱粒子位置隨時間的波動與位置分布
  8. evaluate_and_plot()     - 畫訓練曲線、軌跡熵產生、和解析解對比

使用的庫：
  numpy      - Langevin 模擬、矩陣運算
  scipy      - 解 Lyapunov 方程
  torch      - MLP 定義、自動微分、訓練
  matplotlib - 畫圖

"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.linalg import solve_continuous_lyapunov
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────
# 超參數設定
# ──────────────────────────────────────────────

N_BEADS   = 2        # 粒子數
N_TRJ     = 200      # 軌跡條數
N_STEP    = 5000     # 每條軌跡的步數
DT        = 1e-2     # 時間步長 Δτ
TH        = 10.0     # 高溫熱浴溫度
TC        = 1.0      # 低溫熱浴溫度（Tc/Th = 0.1）
K         = 1.0      # 彈簧常數
GAMMA     = 1.0      # Stokes 摩擦係數
KB        = 1.0      # Boltzmann 常數（無量綱化設為 1）

HIDDEN    = 64       # 每個隱藏層的神經元數
N_LAYERS  = 3        # 隱藏層數
EPOCHS    = 20       # 訓練 epoch 數
BATCH     = 1024     # batch size
LR        = 2e-4     # Adam learning rate
SEED      = 42       # 隨機種子


# ──────────────────────────────────────────────
# 1. 模擬 Bead-Spring 軌跡
# ──────────────────────────────────────────────

def build_A_matrix(n, k, gamma):
    """建立 Langevin 方程式的係數矩陣 A（tri-diagonal）。
    dx/dt = A·x + noise
    A_ij = (-2δ_ij + δ_{i,j+1} + δ_{i+1,j}) * k/gamma
    """
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = -2.0
        if i > 0:
            A[i, i-1] = 1.0
        if i < n - 1:
            A[i, i+1] = 1.0
    return A * k / gamma


def get_temperatures(n, th, tc):
    """各粒子溫度從 Th 線性遞減到 Tc。"""
    return np.linspace(th, tc, n)


def diffusion_matrix(n, th, tc, gamma, kb):
    """擴散矩陣 D = diag(k_B T_i / gamma)。
    Langevin 噪聲滿足 <ξ_i(t)ξ_j(t')> = 2 D_ij δ(t-t')。
    """
    temps = get_temperatures(n, th, tc)
    return np.diag(kb * temps / gamma)


def simulate_bead_spring(n_beads, n_trj, n_step, dt, th, tc, k, gamma, kb, seed=42):
    """Euler-Maruyama 方法模擬 Langevin 方程式。
    回傳 shape: (n_trj, n_step, n_beads)
    """
    rng = np.random.default_rng(seed)
    A = build_A_matrix(n_beads, k, gamma)
    temps = get_temperatures(n_beads, th, tc)
    noise_std = np.sqrt(2.0 * kb * temps / gamma * dt)  # 每步噪聲標準差

    x = rng.normal(0, 1, (n_trj, n_beads))
    trajs = np.zeros((n_trj, n_step, n_beads))

    for t in range(n_step):
        # Euler–Maruyama method
        # x_{t+1} = x_t + A·x_t·dt + sqrt(2k_bT/γ·dt)⊙ z,   z ~ N(0, I) 
        drift = (A @ x.T).T * dt
        noise = rng.normal(0, 1, x.shape) * noise_std
        x = x + drift + noise
        trajs[:, t, :] = x

    return trajs.astype(np.float32)


# ──────────────────────────────────────────────
# 2. Dataset：切成 (s_t, s_{t+1}) pairs
# ──────────────────────────────────────────────

class BeadSpringDataset(Dataset):
    """把形狀 (n_trj, n_step, n_beads) 的軌跡攤開成 (s_t, s_{t+1}) pairs。
    每筆資料是長度 2*n_beads 的向量：[s_t, s_{t+1}] 拼接。
    """
    def __init__(self, trajs):
        s_t   = trajs[:, :-1, :]  # (n_trj, n_step-1, n_beads)
        s_tp1 = trajs[:, 1:,  :]
        s_t   = s_t.reshape(-1, trajs.shape[2])
        s_tp1 = s_tp1.reshape(-1, trajs.shape[2])
        self.data = torch.from_numpy(np.concatenate([s_t, s_tp1], axis=1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ──────────────────────────────────────────────
# 3. MLP 網路架構
# ──────────────────────────────────────────────

class MLP(nn.Module):
    """3 個隱藏層的 MLP，激活函數 ReLU，輸出純量（無激活）。
    輸入維度：2 * n_beads（兩個狀態拼接），輸出維度：1
    """
    def __init__(self, n_beads, hidden=64, n_layers=3):
        super().__init__()
        layers = []
        in_dim = 2 * n_beads
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        layers.append(nn.Linear(hidden, 1))  # 最後一層不加 ReLU
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # 輸出 shape: (batch,)


# ──────────────────────────────────────────────
# 4. NEEP Loss
# ──────────────────────────────────────────────

def compute_delta_s(model, batch):
    """計算每個 pair 的隨機熵產生 ΔS_θ。
    batch shape: (B, 2*n_beads)，前半 s_t，後半 s_{t+1}
    ΔS_θ(s_t, s_{t+1}) = h_θ(s_t, s_{t+1}) - h_θ(s_{t+1}, s_t)
    """
    n = batch.shape[1] // 2
    s_t   = batch[:, :n]  # 前半 s_t
    s_tp1 = batch[:, n:]  # 後半 s_{t+1}
    forward  = torch.cat([s_t,   s_tp1], dim=1)  # 正向
    backward = torch.cat([s_tp1, s_t  ], dim=1)  # 逆向
    return model(forward) - model(backward)


def neep_loss(delta_s):
    """NEEP 目標函數取負值（optimizer 做最小化）。
    J(θ) = E[ΔS_θ - exp(-ΔS_θ)]
    loss = -J(θ)
    """
    return -(delta_s - torch.exp(-delta_s)).mean()


# ──────────────────────────────────────────────
# 5. 訓練迴圈
# ──────────────────────────────────────────────

def train(model, loader, epochs, lr):
    """Adam optimizer 訓練。
    每個 epoch 回傳平均 σ̂ = <ΔS_θ> / Δτ（熵產生率估計值）。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ep_rates = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_ds = 0.0
        total_n  = 0

        for batch in loader:
            optimizer.zero_grad()
            delta_s = compute_delta_s(model, batch)
            loss    = neep_loss(delta_s)
            loss.backward()
            optimizer.step()
            total_ds += delta_s.detach().sum().item()
            total_n  += len(batch)

        # <ΔS_θ> 是「每步」熵產生，除以 Δτ 得到「每單位時間」的熵產生率
        sigma_hat = total_ds / total_n / DT
        ep_rates.append(sigma_hat)
        print(f"Epoch {epoch:02d}/{epochs}  sigma_hat = {sigma_hat:.4f}")

    return ep_rates


# ──────────────────────────────────────────────
# 6. 解析解（已修正：標準 OU 過程穩態熵產生率）
# ──────────────────────────────────────────────

def analytical_ep_rate(n_beads, th, tc, k, gamma, kb):
    """計算線性 Langevin（multivariate Ornstein-Uhlenbeck）的穩態熵產生率 σ̇。

    系統：dx/dt = A·x + ξ，<ξ_i(t)ξ_j(t')> = 2 D_ij δ(t-t')

    標準推導步驟：
      (1) 解 Lyapunov 方程求穩態協方差 C：  A·C + C·A^T + 2D = 0
      (2) 計算機率流矩陣：                  Ω = A·C + D
          （Ω 的反對稱部分代表不可逆的穩態機率流；
            若系統處於平衡態則 Ω 為對稱，σ̇ = 0）
      (3) 熵產生率： σ̇ = Tr(Ω^T D^{-1} Ω C^{-1})

    當 Tc/Th = 1（兩端等溫）時，系統回到平衡態，σ̇ → 0。
    """
    A = build_A_matrix(n_beads, k, gamma)
    D = diffusion_matrix(n_beads, th, tc, gamma, kb)

    # (1) 解 Lyapunov 方程  A C + C A^T = -2D
    C = solve_continuous_lyapunov(A, -2.0 * D)

    # (2) 穩態機率流矩陣
    Omega = A @ C + D

    # (3) 熵產生率
    D_inv = np.linalg.inv(D)
    C_inv = np.linalg.inv(C)
    sigma_dot = np.trace(Omega.T @ D_inv @ Omega @ C_inv)

    return float(sigma_dot)


# ──────────────────────────────────────────────
# 7. 評估與畫圖
# ──────────────────────────────────────────────

def compute_trajectory_ep(model, traj):
    """計算單條軌跡的累積熵產生 S(τ)。traj shape: (n_step, n_beads)"""
    model.eval()
    with torch.no_grad():
        s_t   = torch.from_numpy(traj[:-1])
        s_tp1 = torch.from_numpy(traj[1:])
        batch = torch.cat([s_t, s_tp1], dim=1)
        delta_s = compute_delta_s(model, batch).numpy()
    return np.cumsum(delta_s)


def plot_trajectories(trajs, out="bead_trajectories.png"):
    """畫最熱（bead 0）和最冷（bead N-1）粒子的位置隨時間波動，以及位置分布。"""
    temps = get_temperatures(N_BEADS, TH, TC)
    time_axis = np.arange(1, trajs.shape[1]+1) * DT
    beads = [(0, 'tab:red', 'Hot'), (N_BEADS-1, 'tab:blue', 'Cold')]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle(f"Bead-Spring trajectories (N={N_BEADS})", fontsize=13)

    # ── 圖1：單條軌跡的位置 x_i(τ) ──
    ax = axes[0]
    for i, c, name in beads:
        ax.plot(time_axis, trajs[0, :, i], color=c, lw=0.6, alpha=0.8,
                label=rf"{name} bead $i={i}$ ($T={temps[i]:g}$)")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$x_i(\tau)$")
    ax.set_title("Position vs time (trajectory 0)")
    ax.legend(loc='upper right')

    # ── 圖2：所有軌跡的位置分布 ──
    ax = axes[1]
    for i, c, name in beads:
        ax.hist(trajs[:, :, i].ravel(), bins=80, density=True, color=c, alpha=0.5,
                label=rf"{name}  std={trajs[:, :, i].std():.2f}")
    ax.set_xlabel(r"$x_i$")
    ax.set_ylabel("Density")
    ax.set_title("Position distribution (all trajectories)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"軌跡圖已儲存為 {out}")


def evaluate_and_plot(model, trajs, ep_rates, sigma_analytical):
    """畫三張圖：訓練曲線、單條軌跡累積熵、熵產生率 vs Tc/Th。"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"NEEP - Bead-Spring Model (N={N_BEADS})", fontsize=13)

    # ── 圖1：訓練曲線（σ̂ vs epoch） ──
    ax = axes[0]
    ax.plot(range(1, len(ep_rates)+1), ep_rates, marker='o', ms=4, label=r'$\hat{\sigma}_\theta$')
    ax.axhline(sigma_analytical, color='red', linestyle='--', label=r'$\dot{\sigma}$ (analytical)')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Entropy production rate")
    ax.set_title("Training curve")
    ax.legend()

    # ── 圖2：單條軌跡的累積熵產生 ──
    ax = axes[1]
    cum_ep = compute_trajectory_ep(model, trajs[0])
    time_axis = np.arange(1, len(cum_ep)+1) * DT
    ax.plot(time_axis, cum_ep, label=r'$S_\theta(\tau)$ (NEEP)', alpha=0.8)
    ax.plot(time_axis, sigma_analytical * time_axis, 'r--', label=r'$\dot{\sigma}\cdot\tau$ (analytical)')
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("Cumulative EP")
    ax.set_title("Single trajectory EP")
    ax.legend()

    # ── 圖3：不同 Tc/Th 下的解析解 σ̇ ──
    ax = axes[2]
    tc_th_ratios = np.linspace(0.1, 1.0, 10)
    analytical_vals = [analytical_ep_rate(N_BEADS, TH, r*TH, K, GAMMA, KB) for r in tc_th_ratios]
    ax.plot(tc_th_ratios, analytical_vals, 'b-', marker='s', ms=4, label=r'$\dot{\sigma}$ (analytical)')
    ax.scatter([TC/TH], [sigma_analytical], color='red', zorder=5, s=60,
               label=r'Training point ($T_c/T_h=0.1$)')
    ax.set_xlabel(r"$T_c / T_h$")
    ax.set_ylabel("Entropy production rate")
    ax.set_title(r"$\dot{\sigma}$ vs $T_c/T_h$")
    ax.legend()

    plt.tight_layout()
    plt.savefig("neep_results.png", dpi=120)
    print("\n圖片已儲存為 neep_results.png")
    plt.show()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 50)
    print("Step 1: Simulating Bead-Spring trajectories...")
    print("=" * 50)
    trajs = simulate_bead_spring(N_BEADS, N_TRJ, N_STEP, DT, TH, TC, K, GAMMA, KB, SEED)
    print(f"Trajectory shape: {trajs.shape}  (n_trj, n_step, n_beads)")
    plot_trajectories(trajs)

    print("\n" + "=" * 50)
    print("Step 2: Building Dataset and DataLoader...")
    print("=" * 50)
    n_train = int(N_TRJ * 0.9)
    train_dataset = BeadSpringDataset(trajs[:n_train])
    test_dataset  = BeadSpringDataset(trajs[n_train:])
    train_loader  = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    print(f"Train pairs: {len(train_dataset):,}  |  Test pairs: {len(test_dataset):,}")

    print("\n" + "=" * 50)
    print("Step 3: Building MLP...")
    print("=" * 50)
    model = MLP(N_BEADS, hidden=HIDDEN, n_layers=N_LAYERS)
    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"Total parameters: {n_params:,}")

    print("\n" + "=" * 50)
    print("Step 4: Training NEEP...")
    print("=" * 50)
    ep_rates = train(model, train_loader, EPOCHS, LR)

    print("\n" + "=" * 50)
    print("Step 5: Evaluating...")
    print("=" * 50)
    sigma_analytical = analytical_ep_rate(N_BEADS, TH, TC, K, GAMMA, KB)
    print(f"Analytical sigma_dot = {sigma_analytical:.4f}")
    print(f"NEEP estimate        = {ep_rates[-1]:.4f}")
    print(f"Relative error       = {abs(ep_rates[-1]-sigma_analytical)/sigma_analytical*100:.1f}%")

    evaluate_and_plot(model, trajs, ep_rates, sigma_analytical)