#!/usr/bin/env python3
"""Strict JAX port of `code/models/ltc.jl`.

This file mirrors the current Julia LTC implementation as closely as possible:

    tau(x, I) = tau0 + softplus(W_tau x + U_tau I + b_tau)
    f(x, I) = tanh(W_f x + U_f I + b_f)
    dx/dt = -x / tau(x, I) + f(x, I)

The JAX side also keeps the same conceptual interfaces as Julia:

    flatten(cell)
    unflatten(p_flat, input_dim, hidden_dim)
    ltc_rollout(cell, x0, I_seq, t_span, t_save)
    ltc_rollout_p(p_flat, x0, I_seq, t_span, t_save, input_dim, hidden_dim)

Examples
--------
Run a random rollout:

    conda run --no-capture-output -n jax python code/jax/jax_ltc.py --zeta 0.3

Run with Julia-exported LTC parameters and compare with the Julia reference:

    julia export_for_pytorch_jax.jl
    conda run --no-capture-output -n jax python code/jax/jax_ltc.py \
        --zeta 0.3 \
        --params data/npy/ltc_zeta0p3_params.npy \
        --check-ref
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import NamedTuple

# The local JAX Metal backend is unstable on this machine. Default to CPU
# unless the caller explicitly selects another platform.
if "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "npy"


class LTCCell(NamedTuple):
    """Python counterpart of Julia's `LTCCell{T}`."""

    W_tau: jax.Array
    U_tau: jax.Array
    b_tau: jax.Array
    tau0: jax.Array
    W_f: jax.Array
    U_f: jax.Array
    b_f: jax.Array


def to_tag(zeta: str) -> str:
    """Convert `0.3` to `0p3` for filenames."""

    return zeta.replace(".", "p")


def as_float32(array: np.ndarray | jax.Array) -> np.ndarray:
    """Convert external arrays to float32, matching Julia exports."""

    return np.asarray(array, dtype=np.float32)


def softplus(x: jax.Array) -> jax.Array:
    """Mirror Julia's `log(one(x) + exp(x))` implementation."""

    return jnp.log(jnp.ones_like(x) + jnp.exp(x))


def init_ltc_cell(
    key: jax.Array,
    input_dim: int,
    hidden_dim: int,
    tau0: float = 1.0,
) -> LTCCell:
    """Mirror `LTCCell(input_dim, hidden_dim; tau0=...)` from Julia."""

    scale = 0.1
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return LTCCell(
        W_tau=scale * jax.random.normal(k1, (hidden_dim, hidden_dim), dtype=jnp.float32),
        U_tau=scale * jax.random.normal(k2, (hidden_dim, input_dim), dtype=jnp.float32),
        b_tau=jnp.zeros((hidden_dim,), dtype=jnp.float32),
        tau0=jnp.full((hidden_dim,), tau0, dtype=jnp.float32),
        W_f=scale * jax.random.normal(k3, (hidden_dim, hidden_dim), dtype=jnp.float32),
        U_f=scale * jax.random.normal(k4, (hidden_dim, input_dim), dtype=jnp.float32),
        b_f=jnp.zeros((hidden_dim,), dtype=jnp.float32),
    )


def tau(cell: LTCCell, x: jax.Array, inputs: jax.Array) -> jax.Array:
    """Mirror Julia's `tau(cell, x, I)`."""

    return cell.tau0 + softplus(cell.W_tau @ x + cell.U_tau @ inputs + cell.b_tau)


def f_drift(cell: LTCCell, x: jax.Array, inputs: jax.Array) -> jax.Array:
    """Mirror Julia's `f_drift(cell, x, I)`."""

    return jnp.tanh(cell.W_f @ x + cell.U_f @ inputs + cell.b_f)


def ltc_dynamics(x: jax.Array, cell: LTCCell, inputs: jax.Array) -> jax.Array:
    """Mirror Julia's `ltc_dynamics(x, cell, I)`."""

    tau_val = tau(cell, x, inputs)
    f_val = f_drift(cell, x, inputs)
    return -x / tau_val + f_val


def julia_vec(matrix: jax.Array) -> jax.Array:
    """Flatten a matrix the way Julia's `vec` does: column-major."""

    return jnp.ravel(matrix.T)


def julia_reshape(vector: jax.Array, rows: int, cols: int) -> jax.Array:
    """Reshape a flat vector the way Julia does: fill by columns first."""

    return jnp.reshape(vector, (cols, rows)).T


def flatten(cell: LTCCell) -> jax.Array:
    """Mirror Julia's `flatten(cell)` ordering exactly."""

    return jnp.concatenate(
        (
            julia_vec(cell.W_tau),
            julia_vec(cell.U_tau),
            cell.b_tau,
            cell.tau0,
            julia_vec(cell.W_f),
            julia_vec(cell.U_f),
            cell.b_f,
        )
    )


def unflatten(p_flat: np.ndarray | jax.Array, input_dim: int, hidden_dim: int) -> LTCCell:
    """Mirror Julia's `unflatten(p, input_dim, hidden_dim)` exactly."""

    p = jnp.asarray(p_flat, dtype=jnp.float32).reshape(-1)
    h, i = hidden_dim, input_dim
    idx = 0

    W_tau = julia_reshape(p[idx : idx + h * h], h, h)
    idx += h * h
    U_tau = julia_reshape(p[idx : idx + h * i], h, i)
    idx += h * i
    b_tau = p[idx : idx + h]
    idx += h
    tau0 = p[idx : idx + h]
    idx += h
    W_f = julia_reshape(p[idx : idx + h * h], h, h)
    idx += h * h
    U_f = julia_reshape(p[idx : idx + h * i], h, i)
    idx += h * i
    b_f = p[idx : idx + h]

    return LTCCell(
        W_tau=W_tau,
        U_tau=U_tau,
        b_tau=b_tau,
        tau0=tau0,
        W_f=W_f,
        U_f=U_f,
        b_f=b_f,
    )


def input_at_time(t: jax.Array, t_grid: jax.Array, input_seq: jax.Array) -> jax.Array:
    """Mirror Julia's `idx = clamp(searchsortedfirst(t_grid, t), 1, T)`."""

    idx = jnp.clip(jnp.searchsorted(t_grid, t, side="left"), 0, input_seq.shape[0] - 1)
    return input_seq[idx]


def ltc_rollout_p(
    p_flat: np.ndarray | jax.Array,
    x0: np.ndarray | jax.Array,
    I_seq: np.ndarray | jax.Array,
    t_span: tuple[float, float],
    t_save: np.ndarray | jax.Array,
    input_dim: int,
    hidden_dim: int,
) -> jax.Array:
    """Flat-parameter rollout, matching Julia's `ltc_rollout_p` entry point."""

    p_flat = jnp.asarray(p_flat, dtype=jnp.float32)
    x0 = jnp.asarray(x0, dtype=jnp.float32)
    I_seq = jnp.asarray(I_seq, dtype=jnp.float32)
    t_save = jnp.asarray(t_save, dtype=jnp.float32)

    T = I_seq.shape[0]
    t_grid = jnp.linspace(jnp.float32(t_span[0]), jnp.float32(t_span[1]), T, dtype=jnp.float32)

    def ode_rhs(x: jax.Array, t: jax.Array, p: jax.Array) -> jax.Array:
        cell = unflatten(p, input_dim, hidden_dim)
        inputs = input_at_time(t, t_grid, I_seq)
        return ltc_dynamics(x, cell, inputs)

    return odeint(ode_rhs, x0, t_save, p_flat, rtol=1e-6, atol=1e-6)


def ltc_rollout(
    cell: LTCCell,
    x0: np.ndarray | jax.Array,
    I_seq: np.ndarray | jax.Array,
    t_span: tuple[float, float],
    t_save: np.ndarray | jax.Array,
) -> jax.Array:
    """Struct-style rollout, matching Julia's `ltc_rollout(cell, ...)`."""

    p_flat = flatten(cell)
    return ltc_rollout_p(
        p_flat,
        x0,
        I_seq,
        t_span,
        t_save,
        input_dim=I_seq.shape[1],
        hidden_dim=x0.shape[0],
    )


def readout(hidden_traj: jax.Array, output_dim: int = 2) -> jax.Array:
    """Match the Julia training script's `pred[:, 1:INPUT_DIM]` readout."""

    return hidden_traj[:, :output_dim]


def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    """Plain MSE helper."""

    return float(np.mean((pred - target) ** 2))


def load_ltc_cell(path: Path, input_dim: int, hidden_dim: int) -> LTCCell:
    """Load a Julia-compatible flat `.npy` vector or a named `.npz` archive."""

    if path.suffix == ".npy":
        return unflatten(np.load(path), input_dim, hidden_dim)

    if path.suffix == ".npz":
        data = np.load(path)
        return LTCCell(
            W_tau=jnp.asarray(as_float32(data["W_tau"])),
            U_tau=jnp.asarray(as_float32(data["U_tau"])),
            b_tau=jnp.asarray(as_float32(data["b_tau"])),
            tau0=jnp.asarray(as_float32(data["tau0"])),
            W_f=jnp.asarray(as_float32(data["W_f"])),
            U_f=jnp.asarray(as_float32(data["U_f"])),
            b_f=jnp.asarray(as_float32(data["b_f"])),
        )

    raise ValueError(f"Unsupported parameter file format: {path}")


def load_dataset(zeta: str, split: str = "test", noisy: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Load the Julia-exported oscillator data."""

    tag = to_tag(zeta)
    suffix = "_noisy" if noisy else ""
    t = as_float32(np.load(DATA_DIR / f"oscillator_zeta{tag}_t_{split}.npy"))
    x = as_float32(np.load(DATA_DIR / f"oscillator_zeta{tag}_x_{split}{suffix}.npy"))
    return t, x


def load_reference() -> tuple[np.ndarray, np.ndarray]:
    """Load the Julia LTC alignment reference exported to `.npy`."""

    ref_pred = as_float32(np.load(DATA_DIR / "reference_pred_zeta0p3.npy"))
    ref_t = as_float32(np.load(DATA_DIR / "reference_t_zeta0p3.npy"))
    return ref_pred, ref_t


def verify_against_reference(
    pred: np.ndarray,
    ref_pred: np.ndarray,
    tol: float = 1.0e-4,
) -> tuple[float, bool]:
    """Reference check against Julia output.

    The remaining error floor is mostly driven by solver differences:
    Julia uses `Tsit5()` while this JAX port currently uses
    `jax.experimental.ode.odeint`.
    """

    max_err = float(np.max(np.abs(pred - ref_pred)))
    return max_err, max_err < tol


def parse_args() -> argparse.Namespace:
    """CLI for running or validating the strict JAX LTC port."""

    parser = argparse.ArgumentParser(description="Strict JAX port of code/models/ltc.jl.")
    parser.add_argument("--zeta", default="0.3", help="Damping ratio tag, e.g. 0.3 or 0p3.")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--noisy", action="store_true", help="Use the noisy exported split.")
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--input-dim", type=int, default=2)
    parser.add_argument("--tau0", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--params", type=Path, help="Optional Julia-compatible LTC param export.")
    parser.add_argument("--check-ref", action="store_true", help="Compare against Julia LTC reference output.")
    parser.add_argument(
        "--tol",
        type=float,
        default=1.0e-4,
        help="Alignment tolerance for --check-ref. Keep 1e-4 for strict checking; relax if comparing across solvers.",
    )
    parser.add_argument("--out", type=Path, help="Optional output `.npy` path for the readout trajectory.")
    return parser.parse_args()


def main() -> None:
    """Script entry point."""

    args = parse_args()
    t_eval, x_seq = load_dataset(args.zeta, split=args.split, noisy=args.noisy)
    t_span = (float(t_eval[0]), float(t_eval[-1]))

    if args.params is not None:
        cell = load_ltc_cell(args.params, input_dim=args.input_dim, hidden_dim=args.hidden_dim)
        print(f"Loaded LTC params from {args.params}")
    else:
        key = jax.random.PRNGKey(args.seed)
        cell = init_ltc_cell(key, input_dim=args.input_dim, hidden_dim=args.hidden_dim, tau0=args.tau0)
        print("Initialized random LTCCell.")

    x0 = jnp.zeros((args.hidden_dim,), dtype=jnp.float32)
    hidden_traj = ltc_rollout(cell, x0, x_seq, t_span, t_eval)
    pred = np.asarray(readout(hidden_traj, output_dim=args.input_dim), dtype=np.float32)

    print(f"zeta={args.zeta} split={args.split} noisy={args.noisy}")
    print(f"param_count={int(flatten(cell).size)}")
    print(f"input shape={x_seq.shape} hidden shape={tuple(hidden_traj.shape)} readout shape={pred.shape}")
    print(f"MSE against provided target sequence: {mse_loss(pred, x_seq):.8e}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.out, pred)
        print(f"Saved predictions to {args.out}")

    if args.check_ref:
        ref_pred, ref_t = load_reference()
        if pred.shape != ref_pred.shape:
            raise ValueError(f"Prediction shape {pred.shape} does not match reference shape {ref_pred.shape}.")
        if not np.allclose(t_eval, ref_t, atol=1e-6):
            raise ValueError("Loaded times do not match the exported Julia reference time grid.")

        max_err, passed = verify_against_reference(pred, ref_pred, tol=args.tol)
        status = "PASS" if passed else "FAIL"
        print(f"Alignment check: max_err={max_err:.6f} -> {status} (tol={args.tol:g})")


if __name__ == "__main__":
    main()
