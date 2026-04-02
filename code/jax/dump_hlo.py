#!/usr/bin/env python3
"""Dump HLO for the LTC forward and AD pipeline.

This complements `dump_jaxpr.py` by exporting the compiler-facing HLO text for:

- rollout
- loss
- grad(loss)
- one training step
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

if "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

from jax_ltc import flatten, init_ltc_cell, load_dataset, ltc_rollout_p
from train_ltc import adam_step, clip_gradients, init_adam, mse_loss_jax


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results_jax"


def parse_args() -> argparse.Namespace:
    """CLI arguments for HLO export."""

    parser = argparse.ArgumentParser(description="Dump HLO for JAX LTC analysis.")
    parser.add_argument("--zeta", default="0.3", help="Damping ratio tag, e.g. 0.3 or 0p3.")
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--input-dim", type=int, default=2)
    parser.add_argument("--tau0", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Optional output directory. Defaults to results_jax/hlo_zeta<tag>.",
    )
    return parser.parse_args()


def stablehlo_text(fn: Any, *args: Any) -> str:
    """Lower a function and return textual StableHLO / HLO IR."""

    lowered = jax.jit(fn).lower(*args)
    try:
        return lowered.as_text()
    except AttributeError:
        return lowered.compiler_ir(dialect="hlo").as_hlo_text()


def quick_hlo_summary(text: str) -> dict[str, int]:
    """Count a few high-signal compiler-level structures."""

    return {
        "line_count": len(text.splitlines()),
        "while_count": text.count("stablehlo.while") + text.count(" while "),
        "fusion_count": text.count("stablehlo.fusion") + text.count(" fusion"),
        "dot_count": text.count("stablehlo.dot_general") + text.count("dot_general"),
        "tanh_count": text.count("stablehlo.tanh") + text.count("tanh"),
        "exponential_count": text.count("stablehlo.exponential") + text.count("exponential"),
        "log_count": text.count("stablehlo.log") + text.count(" log "),
    }


def main() -> None:
    """Generate HLO artifacts for one zeta setting."""

    args = parse_args()
    tag = args.zeta.replace(".", "p")
    out_dir = args.out_dir or (RESULTS_DIR / f"hlo_zeta{tag}")
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

    def rollout_fn(p_flat: jax.Array) -> jax.Array:
        return ltc_rollout_p(
            p_flat,
            x0,
            x_train,
            t_span_train,
            t_train_jax,
            args.input_dim,
            args.hidden_dim,
        )

    def loss_fn(p_flat: jax.Array) -> jax.Array:
        hidden = rollout_fn(p_flat)
        pred = hidden[:, : args.input_dim]
        return mse_loss_jax(pred, x_train)

    grad_fn = jax.grad(loss_fn)
    loss_and_grad_fn = jax.value_and_grad(loss_fn)

    def train_step(p_flat: jax.Array, state: Any) -> tuple[jax.Array, Any, jax.Array]:
        loss, grads = loss_and_grad_fn(p_flat)
        grads = clip_gradients(grads, args.grad_clip)
        new_params, new_state = adam_step(p_flat, grads, state, lr=args.lr)
        return new_params, new_state, loss

    artifacts: dict[str, tuple[Any, tuple[Any, ...]]] = {
        "rollout": (rollout_fn, (params,)),
        "loss": (loss_fn, (params,)),
        "grad": (grad_fn, (params,)),
        "train_step": (train_step, (params, adam_state)),
    }

    summary: dict[str, Any] = {
        "zeta": args.zeta,
        "input_dim": args.input_dim,
        "hidden_dim": args.hidden_dim,
        "param_count": int(params.size),
        "backend": jax.default_backend(),
        "artifacts": {},
    }

    for name, (fn, fn_args) in artifacts.items():
        text = stablehlo_text(fn, *fn_args)
        path = out_dir / f"{name}.mlir"
        path.write_text(text, encoding="utf-8")
        summary["artifacts"][name] = {
            "path": str(path),
            **quick_hlo_summary(text),
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved HLO artifacts to {out_dir}")
    print(f"Summary -> {summary_path}")
    for name in ("rollout", "loss", "grad", "train_step"):
        artifact = summary["artifacts"][name]
        print(
            f"{name:10s} lines={artifact['line_count']:5d} "
            f"while={artifact['while_count']:2d} fusion={artifact['fusion_count']:2d}"
        )


if __name__ == "__main__":
    main()
