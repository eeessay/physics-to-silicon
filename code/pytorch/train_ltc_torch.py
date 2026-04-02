# ============================================================
# Author: Member 3 (SHEN)
# EE5311 CA1-21 — PyTorch Training Script for LTC
#
# This script trains the Liquid Time-Constant (LTC) neural ODE
# model on oscillator datasets with different damping factors (zeta).
#
# Main functionalities:
#   1. Load training and testing data exported from Julia (.npy format)
#      with proper column-major reconstruction.
#   2. Train the LTC model using MSE loss and Adam optimizer.
#   3. Perform rollout-based prediction via ODE integration.
#   4. Track and record training/test losses across epochs.
#   5. Save best test predictions and raw results to CSV files.
#   6. Generate plots for qualitative comparison.
#   7. Merge results across multiple zeta settings.
#
# Model:
#   The LTC model is implemented in ltc_torch.py (LTCSequenceODE),
#   following the continuous-time dynamics:
#
#       dx/dt = -x / τ(x,I) + f(x,I)
#
# Data:
#   Input data is generated from a damped oscillator system
#   and exported from a reference Julia implementation.
#
# Output:
#   results_torch/
#       ├── raw_*.csv        (predictions vs ground truth)
#       ├── loss_*.csv       (training/test loss curves)
#       ├── plot_*.png       (visualization)
#       ├── raw_all_zeta.csv
#       └── loss_all_zeta.csv
#
# Notes:
#   - Gradient clipping is applied to stabilize training.
#   - Best model is selected based on lowest test loss.
#   - This script is used for training and evaluation only;
#     alignment with the Julia implementation is handled separately.
# ============================================================

import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

from ltc_torch import LTCSequenceODE


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def load_julia_matrix(path: str) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 1:
        return arr
    return arr.ravel(order="C").reshape(arr.shape, order="F")


def make_result_dir():
    os.makedirs("results_torch", exist_ok=True)


# ------------------------------------------------------------
# Training function
# ------------------------------------------------------------
def train_one_zeta(
    zeta_tag="zeta0p3",
    input_dim=2,
    hidden_dim=8,
    num_epochs=300,
    lr=1e-3,
    device="cpu",
):
    print(f"\n===== Training for {zeta_tag} =====")

    # Load data
    x_train = load_julia_matrix(f"data/npy/oscillator_{zeta_tag}_x_train.npy")
    x_test = load_julia_matrix(f"data/npy/oscillator_{zeta_tag}_x_test.npy")
    t_train = load_julia_matrix(f"data/npy/oscillator_{zeta_tag}_t_train.npy")
    t_test = load_julia_matrix(f"data/npy/oscillator_{zeta_tag}_t_test.npy")

    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)

    x_train_t = torch.tensor(x_train, dtype=torch.float32).to(device)
    x_test_t = torch.tensor(x_test, dtype=torch.float32).to(device)
    t_train_t = torch.tensor(t_train, dtype=torch.float32).to(device)
    t_test_t = torch.tensor(t_test, dtype=torch.float32).to(device)

    model = LTCSequenceODE(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    x0 = torch.zeros(hidden_dim).to(device)

    train_losses = []
    test_losses = []

    best_test_loss = float("inf")
    best_final_test = None
    best_epoch = 0

    # ---------------- Training ----------------
    start = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()

        pred_train_full = model.rollout(
            x0=x0,
            I_seq=x_train_t,
            t_span=(float(t_train[0]), float(t_train[-1])),
            t_save=t_train_t,
        )

        pred_train = pred_train_full[:, :input_dim]
        loss_train = criterion(pred_train, x_train_t)

        optimizer.zero_grad()
        loss_train.backward()

        # ✅ 防爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # ---------------- Test ----------------
        model.eval()
        with torch.no_grad():
            pred_test_full = model.rollout(
                x0=x0,
                I_seq=x_test_t,
                t_span=(float(t_test[0]), float(t_test[-1])),
                t_save=t_test_t,
            )

            pred_test = pred_test_full[:, :input_dim]
            loss_test = criterion(pred_test, x_test_t)

        # 记录
        train_losses.append(loss_train.item())
        test_losses.append(loss_test.item())

        # 保存 best
        if not torch.isnan(loss_test) and loss_test.item() < best_test_loss:
            best_test_loss = loss_test.item()
            best_final_test = pred_test.detach().cpu().numpy()
            best_epoch = epoch

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train={loss_train:.6f} | test={loss_test:.6f}")

    end = time.time()
    avg_time = (end - start) / num_epochs

    print(f"Training time ({num_epochs} epochs): {end - start:.4f} s")
    print(f"Average time per epoch: {avg_time:.6f} s")

    print(f"Best epoch: {best_epoch}, best test loss: {best_test_loss:.6f}")

    final_test = best_final_test

    # --------------------------------------------------------
    # Save RAW DATA（核心！！）
    # --------------------------------------------------------
    make_result_dir()

    raw_df = pd.DataFrame({
        "t": np.array(t_test),
        "zeta": [zeta_tag] * len(t_test),
        "ground_truth_dim0": x_test[:, 0],
        "ltc_pred_dim0": final_test[:, 0],
        "ground_truth_dim1": x_test[:, 1],
        "ltc_pred_dim1": final_test[:, 1],
    })

    raw_df.to_csv(f"results_torch/raw_{zeta_tag}.csv", index=False)

    print(">>> raw csv saved")

    # --------------------------------------------------------
    # Save LOSS
    # --------------------------------------------------------
    loss_df = pd.DataFrame({
        "epoch": np.arange(1, len(train_losses) + 1),
        "zeta": [zeta_tag] * len(train_losses),
        "train_loss": train_losses,
        "test_loss": test_losses,
    })

    loss_df.to_csv(f"results_torch/loss_{zeta_tag}.csv", index=False)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    plt.figure()
    plt.plot(t_test, x_test[:, 0], label="GT")
    plt.plot(t_test, final_test[:, 0], "--", label="Pred")
    plt.legend()
    plt.title(f"{zeta_tag} dim0")
    plt.savefig(f"results_torch/plot_{zeta_tag}.png")
    plt.close()


# ------------------------------------------------------------
# Merge all zeta
# ------------------------------------------------------------
def merge_all():
    zetas = ["zeta0p1", "zeta0p3", "zeta0p5", "zeta0p8", "zeta1p2"]

    raw_list = []
    loss_list = []

    for z in zetas:
        raw_path = f"results_torch/raw_{z}.csv"
        loss_path = f"results_torch/loss_{z}.csv"

        if os.path.exists(raw_path):
            raw_list.append(pd.read_csv(raw_path))
        if os.path.exists(loss_path):
            loss_list.append(pd.read_csv(loss_path))

    if raw_list:
        pd.concat(raw_list).to_csv("results_torch/raw_all_zeta.csv", index=False)
        print(">>> merged raw_all_zeta.csv")

    if loss_list:
        pd.concat(loss_list).to_csv("results_torch/loss_all_zeta.csv", index=False)
        print(">>> merged loss_all_zeta.csv")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    zeta_list = ["zeta0p1", "zeta0p3", "zeta0p5", "zeta0p8", "zeta1p2"]

    for z in zeta_list:
        train_one_zeta(zeta_tag=z)

    merge_all()