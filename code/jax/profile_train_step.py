#!/usr/bin/env python3
"""Profile the JAX LTC training step at the AD/compile boundary.

This script focuses on the bottom-layer comparison point we care about:

- JAX tracing/lowering cost
- XLA compilation cost
- first execution latency
- steady-state execution latency

It uses the exact same LTC forward, loss, gradient, clipping, and Adam update
stack as `train_ltc.py`, but only measures one `jit(train_step)` pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

if "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np

from jax_ltc import flatten, init_ltc_cell, load_dataset, ltc_rollout_p
from train_ltc import adam_step, clip_gradients, init_adam, mse_loss_jax


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results_jax"


def parse_args() -> argparse.Namespace:
    """CLI arguments for train-step profiling."""

    parser = argparse.ArgumentParser(description="Profile JAX LTC train_step.")
    parser.add_argument("--zeta", default="0.3", help="Damping ratio tag, e.g. 0.3 or 0p3.")
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--input-dim", type=int, default=2)
    parser.add_argument("--tau0", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eager-iters", type=int, default=5)
    parser.add_argument("--steady-iters", type=int, default=20)
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Optional output directory. Defaults to results_jax/profile_zeta<tag>.",
    )
    return parser.parse_args()


def block_tree(tree: Any) -> Any:
    """Synchronize all arrays in a pytree."""

    return jax.tree_util.tree_map(jax.block_until_ready, tree)


def main() -> None:
    """Build and profile one AD-enabled LTC training step."""

    args = parse_args()
    tag = args.zeta.replace(".", "p")
    out_dir = args.out_dir or (RESULTS_DIR / f"profile_zeta{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    t_train, x_train_np = load_dataset(args.zeta, split="train", noisy=False)
    x_train = jnp.asarray(x_train_np, dtype=jnp.float32)
    t_train_jax = jnp.asarray(t_train, dtype=jnp.float32)
    t_span_train = (float(t_train[0]), float(t_train[-1]))
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

    loss_and_grad = jax.value_and_grad(loss_fn)

    def train_step(p_flat: jax.Array, state: Any) -> tuple[jax.Array, Any, jax.Array]:
        loss, grads = loss_and_grad(p_flat)
        grads = clip_gradients(grads, args.grad_clip)
        new_params, new_state = adam_step(p_flat, grads, state, lr=args.lr)
        return new_params, new_state, loss

    jit_train_step = jax.jit(train_step)

    eager_first_start = time.perf_counter()
    eager_out = train_step(params, adam_state)
    block_tree(eager_out)
    eager_first_sec = time.perf_counter() - eager_first_start

    eager_times: list[float] = []
    p_eager, s_eager, _ = eager_out
    for _ in range(args.eager_iters):
        iter_start = time.perf_counter()
        out = train_step(p_eager, s_eager)
        p_eager, s_eager, eager_loss = out
        block_tree((p_eager, s_eager, eager_loss))
        eager_times.append(time.perf_counter() - iter_start)

    lower_start = time.perf_counter()
    lowered = jit_train_step.lower(params, adam_state)
    lower_sec = time.perf_counter() - lower_start

    compile_start = time.perf_counter()
    compiled = lowered.compile()
    compile_sec = time.perf_counter() - compile_start

    first_exec_start = time.perf_counter()
    first_out = compiled(params, adam_state)
    block_tree(first_out)
    first_execute_sec = time.perf_counter() - first_exec_start

    steady_times: list[float] = []
    p_run, s_run, _ = first_out
    for _ in range(args.steady_iters):
        iter_start = time.perf_counter()
        out = compiled(p_run, s_run)
        p_run, s_run, loss_run = out
        block_tree((p_run, s_run, loss_run))
        steady_times.append(time.perf_counter() - iter_start)

    loss_value = float(loss_run)
    summary = {
        "zeta": args.zeta,
        "backend": jax.default_backend(),
        "input_dim": args.input_dim,
        "hidden_dim": args.hidden_dim,
        "param_count": int(params.size),
        "eager_iters": args.eager_iters,
        "steady_iters": args.steady_iters,
        "eager_first_execute_sec": eager_first_sec,
        "eager_execute_sec_mean": float(np.mean(eager_times)),
        "eager_execute_sec_std": float(np.std(eager_times)),
        "lower_sec": lower_sec,
        "compile_sec": compile_sec,
        "first_execute_sec": first_execute_sec,
        "steady_execute_sec_mean": float(np.mean(steady_times)),
        "steady_execute_sec_std": float(np.std(steady_times)),
        "steady_execute_sec_min": float(np.min(steady_times)),
        "steady_execute_sec_max": float(np.max(steady_times)),
        "last_loss": loss_value,
    }

    summary_path = out_dir / "train_step_timing.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved profile -> {summary_path}")
    print(
        "Timing "
        f"eager_first={eager_first_sec:.6f}s "
        f"eager_mean={summary['eager_execute_sec_mean']:.6f}s "
        f"lower={lower_sec:.6f}s "
        f"compile={compile_sec:.6f}s "
        f"first_execute={first_execute_sec:.6f}s "
        f"steady_mean={summary['steady_execute_sec_mean']:.6f}s"
    )


if __name__ == "__main__":
    main()
