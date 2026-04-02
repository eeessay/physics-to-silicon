#!/usr/bin/env python3
"""Visualize the damped-oscillator dataset from exported NumPy files.

Run from repo root with the conda `jax` environment:

    conda run --no-capture-output -n jax python code/jax/visualize_dataset.py

Outputs:
    results/dataset_overview.png
    results/dataset_zeta0p1.png
    results/dataset_zeta0p3.png
    ...
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "npy"
RESULTS_DIR = ROOT / "results"
ZETAS = ("0.1", "0.3", "0.5", "0.8", "1.2")


def zeta_tag(zeta: str) -> str:
    return zeta.replace(".", "p")


def load_split(zeta: str) -> dict[str, np.ndarray]:
    tag = zeta_tag(zeta)
    return {
        "t_train": np.load(DATA_DIR / f"oscillator_zeta{tag}_t_train.npy").astype(np.float32),
        "x_train": np.load(DATA_DIR / f"oscillator_zeta{tag}_x_train.npy").astype(np.float32),
        "t_test": np.load(DATA_DIR / f"oscillator_zeta{tag}_t_test.npy").astype(np.float32),
        "x_test": np.load(DATA_DIR / f"oscillator_zeta{tag}_x_test.npy").astype(np.float32),
        "x_train_noisy": np.load(DATA_DIR / f"oscillator_zeta{tag}_x_train_noisy.npy").astype(np.float32),
        "x_test_noisy": np.load(DATA_DIR / f"oscillator_zeta{tag}_x_test_noisy.npy").astype(np.float32),
    }


def plot_single_zeta(zeta: str) -> Path:
    d = load_split(zeta)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(d["t_train"], d["x_train"][:, 0], label="train x(t)", color="tab:blue", linewidth=2)
    axes[0].plot(d["t_train"], d["x_train"][:, 1], label="train v(t)", color="tab:green", linewidth=2)
    axes[0].plot(d["t_test"], d["x_test"][:, 0], label="test x(t)", color="tab:red", linestyle="--", linewidth=2)
    axes[0].plot(d["t_test"], d["x_test"][:, 1], label="test v(t)", color="tab:orange", linestyle="--", linewidth=2)
    axes[0].set_title(f"zeta={zeta} clean trajectories")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("value")
    axes[0].legend()

    axes[1].plot(d["x_train"][:, 0], d["x_train"][:, 1], label="train phase", color="tab:purple", linewidth=2)
    axes[1].plot(d["x_test"][:, 0], d["x_test"][:, 1], label="test phase", color="black", linestyle="--", linewidth=2)
    axes[1].set_title("Phase portrait")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("v")
    axes[1].legend()

    axes[2].plot(d["t_test"], d["x_test"][:, 0], label="clean x(t)", color="tab:blue", linewidth=2)
    axes[2].plot(
        d["t_test"],
        d["x_test_noisy"][:, 0],
        label="noisy x(t)",
        color="tab:red",
        linestyle="--",
        linewidth=2,
    )
    axes[2].set_title("Test clean vs noisy")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("position")
    axes[2].legend()

    fig.tight_layout()
    out_path = RESULTS_DIR / f"dataset_zeta{zeta_tag(zeta)}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_overview() -> Path:
    fig, axes = plt.subplots(len(ZETAS), 1, figsize=(10, 2.4 * len(ZETAS)), sharex=False)

    for ax, zeta in zip(axes, ZETAS):
        d = load_split(zeta)
        ax.plot(d["t_train"], d["x_train"][:, 0], label="train x(t)", color="tab:blue", linewidth=2)
        ax.plot(d["t_test"], d["x_test"][:, 0], label="test x(t)", color="tab:red", linestyle="--", linewidth=2)
        ax.set_title(f"zeta={zeta}")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.legend()

    fig.tight_layout()
    out_path = RESULTS_DIR / "dataset_overview.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating dataset visualizations...")

    for zeta in ZETAS:
        out_path = plot_single_zeta(zeta)
        print(f"  zeta={zeta} -> {out_path.relative_to(ROOT)}")

    overview_path = plot_overview()
    print(f"Saved overview -> {overview_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
