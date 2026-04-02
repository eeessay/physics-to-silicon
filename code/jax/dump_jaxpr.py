#!/usr/bin/env python3
"""Dump JAXPRs for the LTC forward and AD pipeline.

This script is meant for analysis, not training. It generates JAXPR text files
for the key stages of the JAX LTC training stack:

- rollout
- loss
- grad(loss)
- one training step

It also writes a primitive-count summary to JSON so the AD path can be
discussed more concretely in the report.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
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
    """CLI arguments for JAXPR export."""

    parser = argparse.ArgumentParser(description="Dump JAXPRs for JAX LTC analysis.")
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
        help="Optional output directory. Defaults to results_jax/jaxpr_zeta<tag>.",
    )
    return parser.parse_args()


def _walk_jaxpr_like(obj: Any, counter: Counter[str]) -> None:
    """Recursively collect primitive names from nested JAXPR structures."""

    if obj is None:
        return

    if hasattr(obj, "eqns") and hasattr(obj, "constvars"):
        for eqn in obj.eqns:
            counter[str(eqn.primitive)] += 1
            for value in eqn.params.values():
                _walk_jaxpr_like(value, counter)
        return

    if hasattr(obj, "jaxpr"):
        _walk_jaxpr_like(obj.jaxpr, counter)
        return

    if isinstance(obj, dict):
        for value in obj.values():
            _walk_jaxpr_like(value, counter)
        return

    if isinstance(obj, (list, tuple)):
        for value in obj:
            _walk_jaxpr_like(value, counter)


def primitive_summary(closed_jaxpr: Any) -> dict[str, Any]:
    """Create a compact summary of primitives appearing in a JAXPR."""

    counter: Counter[str] = Counter()
    _walk_jaxpr_like(closed_jaxpr, counter)
    return {
        "top_level_eqns": len(closed_jaxpr.jaxpr.eqns),
        "primitive_counts": dict(sorted(counter.items())),
    }


def main() -> None:
    """Generate JAXPR artifacts for one zeta setting."""

    args = parse_args()
    tag = args.zeta.replace(".", "p")
    out_dir = args.out_dir or (RESULTS_DIR / f"jaxpr_zeta{tag}")
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

    rollout_jaxpr = jax.make_jaxpr(rollout_fn)(params)
    loss_jaxpr = jax.make_jaxpr(loss_fn)(params)
    grad_jaxpr = jax.make_jaxpr(grad_fn)(params)
    train_step_jaxpr = jax.make_jaxpr(train_step)(params, adam_state)

    artifacts = {
        "rollout": rollout_jaxpr,
        "loss": loss_jaxpr,
        "grad": grad_jaxpr,
        "train_step": train_step_jaxpr,
    }

    summary: dict[str, Any] = {
        "zeta": args.zeta,
        "input_dim": args.input_dim,
        "hidden_dim": args.hidden_dim,
        "param_count": int(params.size),
        "backend": jax.default_backend(),
        "artifacts": {},
    }

    for name, closed_jaxpr in artifacts.items():
        path = out_dir / f"{name}.txt"
        path.write_text(str(closed_jaxpr), encoding="utf-8")
        summary["artifacts"][name] = {
            "path": str(path),
            **primitive_summary(closed_jaxpr),
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved JAXPR artifacts to {out_dir}")
    print(f"Summary -> {summary_path}")
    for name in ("rollout", "loss", "grad", "train_step"):
        artifact = summary["artifacts"][name]
        print(
            f"{name:10s} top_level_eqns={artifact['top_level_eqns']:4d} "
            f"primitive_types={len(artifact['primitive_counts'])}"
        )


if __name__ == "__main__":
    main()
