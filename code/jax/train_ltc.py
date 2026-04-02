#!/usr/bin/env python3
"""Train the LTC model in JAX with automatic differentiation.

This script mirrors the Julia `train_ltc(zeta)` loop in `code/train.jl`,
but uses JAX `value_and_grad` instead of Zygote.

Default hyperparameters intentionally match the Julia training script:

    EPOCHS     = 300
    INPUT_DIM  = 2
    HIDDEN_DIM = 8
    LR         = 1e-3
    GRAD_CLIP  = 1.0
    SEED       = 42

Examples
--------
Smoke test:

    conda run --no-capture-output -n jax python code/jax/train_ltc.py --zeta 0.3 --epochs 1

Full run:

    conda run --no-capture-output -n jax python code/jax/train_ltc.py --zeta 0.3
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import NamedTuple

# Default to CPU to avoid unstable local Metal backend behavior.
if "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np

from jax_ltc import flatten, init_ltc_cell, load_dataset, ltc_rollout_p


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results_jax"


class AdamState(NamedTuple):
    """Minimal Adam state, mirroring the Julia optimizer structure."""

    t: jnp.int32
    m: jax.Array
    v: jax.Array


def mse_loss_jax(pred: jax.Array, target: jax.Array) -> jax.Array:
    """JAX version of the common MSE loss."""

    return jnp.mean((pred - target) ** 2)


def init_adam(n_params: int) -> AdamState:
    """Initialize first and second moments to zero."""

    zeros = jnp.zeros((n_params,), dtype=jnp.float32)
    return AdamState(t=jnp.int32(0), m=zeros, v=zeros)


def clip_gradients(grads: jax.Array, max_norm: float) -> jax.Array:
    """Global norm clipping, matching the Julia training script."""

    grad_norm = jnp.linalg.norm(grads)
    scale = jnp.minimum(1.0, max_norm / (grad_norm + 1e-12))
    return grads * scale


def adam_step(
    params: jax.Array,
    grads: jax.Array,
    state: AdamState,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[jax.Array, AdamState]:
    """One Adam update step."""

    t = state.t + jnp.int32(1)
    m = beta1 * state.m + (1.0 - beta1) * grads
    v = beta2 * state.v + (1.0 - beta2) * (grads**2)
    m_hat = m / (1.0 - beta1**t)
    v_hat = v / (1.0 - beta2**t)
    params = params - lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return params, AdamState(t=t, m=m, v=v)


def parse_args() -> argparse.Namespace:
    """CLI arguments for JAX LTC training."""

    parser = argparse.ArgumentParser(description="Train the LTC model in JAX.")
    parser.add_argument("--zeta", default="0.3", help="Damping ratio tag, e.g. 0.3 or 0p3.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--input-dim", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--tau0", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--jit-train-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to JIT-compile the full train step. Defaults to True.",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        help="Optional checkpoint prefix. Defaults to results_jax/ltc_zeta<tag>.",
    )
    return parser.parse_args()


def main() -> None:
    """Train one LTC model for one zeta value."""

    args = parse_args()
    tag = args.zeta.replace(".", "p")

    t_train, x_train_np = load_dataset(args.zeta, split="train", noisy=False)
    t_test, x_test_np = load_dataset(args.zeta, split="test", noisy=False)

    x_train = jnp.asarray(x_train_np, dtype=jnp.float32)
    x_test = jnp.asarray(x_test_np, dtype=jnp.float32)
    t_train_jax = jnp.asarray(t_train, dtype=jnp.float32)
    t_test_jax = jnp.asarray(t_test, dtype=jnp.float32)

    t_span_train = (float(t_train[0]), float(t_train[-1]))
    t_span_test = (float(t_test[0]), float(t_test[-1]))
    x0 = jnp.zeros((args.hidden_dim,), dtype=jnp.float32)

    key = jax.random.PRNGKey(args.seed)
    cell = init_ltc_cell(key, input_dim=args.input_dim, hidden_dim=args.hidden_dim, tau0=args.tau0)
    params = flatten(cell)
    adam_state = init_adam(int(params.size))

    def loss_fn(p_flat: jax.Array) -> jax.Array:
        hidden = ltc_rollout_p(
            p_flat,
            x0,
            x_train,
            t_span_train,
            t_train_jax,
            args.input_dim,
            args.hidden_dim,
        )
        pred = hidden[:, : args.input_dim]
        return mse_loss_jax(pred, x_train)

    def val_loss_fn(p_flat: jax.Array) -> jax.Array:
        hidden = ltc_rollout_p(
            p_flat,
            x0,
            x_test,
            t_span_test,
            t_test_jax,
            args.input_dim,
            args.hidden_dim,
        )
        pred = hidden[:, : args.input_dim]
        return mse_loss_jax(pred, x_test)

    loss_and_grad = jax.value_and_grad(loss_fn)

    print(f"Training JAX LTC on zeta={args.zeta} ({int(params.size)} parameters) [AD]")

    best_val_loss = float("inf")
    best_params = np.asarray(params, dtype=np.float32).copy()
    history: list[dict[str, float | int]] = []
    epoch_times: list[float] = []
    compile_sec = 0.0

    def train_step(p_flat: jax.Array, state: AdamState) -> tuple[jax.Array, AdamState, jax.Array, jax.Array]:
        loss, grads = loss_and_grad(p_flat)
        grads = clip_gradients(grads, args.grad_clip)
        grad_norm = jnp.linalg.norm(grads)
        new_params, new_state = adam_step(p_flat, grads, state, lr=args.lr)
        return new_params, new_state, loss, grad_norm

    if args.jit_train_step:
        compile_start = time.perf_counter()
        compiled_train_step = jax.jit(train_step).lower(params, adam_state).compile()
        compile_sec = time.perf_counter() - compile_start
        step_fn = compiled_train_step
    else:
        step_fn = train_step

    total_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        params, adam_state, train_loss, grad_norm = step_fn(params, adam_state)
        params, adam_state, train_loss, grad_norm = jax.block_until_ready((params, adam_state, train_loss, grad_norm))
        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)

        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            train_loss_val = float(train_loss)
            val_loss_val = float(val_loss_fn(params))
            grad_norm_val = float(grad_norm)

            print(
                f"  zeta={args.zeta} epoch {epoch:3d}/{args.epochs} "
                f"train={train_loss_val:.6f} val={val_loss_val:.6f} "
                f"grad_norm={grad_norm_val:.6f} epoch_time={epoch_time:.4f}s"
            )

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss_val,
                    "val_loss": val_loss_val,
                    "grad_norm": grad_norm_val,
                    "epoch_time_sec": epoch_time,
                }
            )

            if val_loss_val < best_val_loss:
                best_val_loss = val_loss_val
                best_params = np.asarray(params, dtype=np.float32).copy()

    total_training_sec = time.perf_counter() - total_start
    first_epoch_sec = epoch_times[0] if epoch_times else 0.0
    steady_epoch_sec = float(np.mean(epoch_times[1:])) if len(epoch_times) > 1 else first_epoch_sec

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_prefix = args.out_prefix or (RESULTS_DIR / f"ltc_zeta{tag}")
    out_prefix = Path(out_prefix)

    np.save(out_prefix.with_suffix(".npy"), best_params)
    with out_prefix.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "zeta": args.zeta,
                "epochs": args.epochs,
                "input_dim": args.input_dim,
                "hidden_dim": args.hidden_dim,
                "lr": args.lr,
                "grad_clip": args.grad_clip,
                "seed": args.seed,
                "jit_train_step": args.jit_train_step,
                "best_val_loss": best_val_loss,
                "backend": jax.default_backend(),
                "compile_sec": compile_sec,
                "total_training_sec": total_training_sec,
                "first_epoch_sec": first_epoch_sec,
                "steady_epoch_sec_mean": steady_epoch_sec,
                "history": history,
            },
            f,
            indent=2,
        )

    print(f"Saved best params -> {out_prefix.with_suffix('.npy')}")
    print(f"Saved metrics     -> {out_prefix.with_suffix('.json')}")
    print(f"Best val loss     -> {best_val_loss:.6f}")
    print(
        "Timing            -> "
        f"compile={compile_sec:.4f}s "
        f"total={total_training_sec:.4f}s "
        f"first_epoch={first_epoch_sec:.4f}s "
        f"steady_mean={steady_epoch_sec:.4f}s"
    )


if __name__ == "__main__":
    main()
