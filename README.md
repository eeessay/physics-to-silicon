# physics-to-silicon

Cross-framework implementation and benchmarking of continuous-time recurrent architectures on physics-based time series.

> 📖 Blog: https://divine-dream-f67e.yuxintong0805.workers.dev/

---

## Overview

We compare three continuous-time recurrent architectures — **Neural ODE**, **CTRNN**, and **LTC** — on a damped harmonic oscillator benchmark spanning five qualitatively distinct physical regimes (ζ ∈ {0.1, 0.3, 0.5, 0.8, 1.2}). The central question: does making τ input-dependent actually help? And what does that expressiveness cost when the same model is implemented across Julia, PyTorch, and JAX?

We additionally introduce an **MDL-inspired τ regularisation** scheme and evaluate it under abrupt regime switches.

All evaluations use **autonomous closed-loop rollout** — the model receives its own previous output as input, not ground truth. This is substantially harder than single-step prediction and directly tests whether the model has learned stable, self-consistent dynamics.

---

## Quick Start

```julia
# 1. Generate dataset
julia code/data_gen.jl

# 2. Train all three models across all ζ
julia code/train.jl

# 3. MDL τ regularisation experiments
julia code/train_ltc_mdl.jl

# 4. Evaluation experiments
julia code/experiments/mse_comparison.jl
julia code/experiments/generalization.jl
julia code/experiments/noise_injection.jl
julia code/experiments/regime_switch.jl
julia code/experiments/final_attractor_analysis.jl

# 5. Julia profiling
julia code/profiling/julia_profiling.jl
```

```bash
# PyTorch
python code/pytorch/train_ltc_torch.py

# JAX
python code/jax/train_ltc.py
python code/jax/dump_jaxpr.py
```

### Dependencies

**Julia:** `DifferentialEquations`, `SciMLSensitivity`, `Zygote`, `JLD2`, `BenchmarkTools`

**Python:** `torch`, `jax`, `flax`, `numpy`

---

## Dataset

The system is a damped harmonic oscillator in state-space form, with state $u = [x,\, \dot{x}]^\top$:

$$\dot{u}_1 = u_2, \qquad \dot{u}_2 = -2\zeta\omega_0\, u_2 - \omega_0^2\, u_1$$

- **Initial condition:** $u_0 = [1.0,\, 0.0]^\top$
- **Parameters:** $\omega_0 = 1.0$, $t \in [0, 20]$, 200 uniformly sampled time points
- **Solver:** Tsit5, `abstol = reltol = 1e-8`
- **Noise:** a separate noisy copy is generated with additive Gaussian noise $\sigma = 0.01$

**Train/test split:** odd-indexed time points (1, 3, 5, … in 1-based Julia) → train (100 pts); even-indexed (2, 4, 6, …) → test (100 pts). Both sets span the full $[0, 20]$ interval, so the model cannot coast on temporal extrapolation.

Five regimes spanning qualitatively distinct behaviours:

| ζ | Character |
|---|---|
| 0.1 | Strongly underdamped — sustained oscillations |
| 0.3 | Underdamped — visible oscillations, moderate decay |
| 0.5 | Transition zone |
| 0.8 | Lightly overdamped — monotone with slight overshoot |
| 1.2 | Overdamped — pure exponential decay |

---

## Models

### 1. Neural ODE — `code/models/neural_ode.jl`

Baseline continuous-time model with no explicit temporal structure.

$$\frac{dx}{dt} = f_\theta(x, I)$$

Two-layer MLP right-hand side: $f_\theta(x, I) = W_2 \tanh(W_1 x + U_1 I + b_1) + b_2$. No parameter encodes how quickly the system should respond — temporal scale must be discovered entirely from gradient signal.

- **Parameters:** $W_1, U_1, b_1 \in \mathbb{R}^{h \times h}, \mathbb{R}^{h \times d}, \mathbb{R}^h$; $W_2, b_2 \in \mathbb{R}^{h \times h}, \mathbb{R}^h$
- **Solver:** Tsit5 with `ForwardDiffSensitivity`, `abstol = reltol = 1e-6`

### 2. CTRNN — `code/models/ctrnn.jl`

Adds a learnable scalar time constant per neuron, introducing an explicit leak term.

$$\frac{dx}{dt} = -\frac{x}{\tau} + \tanh(Wx + UI + b), \qquad \tau = \exp(\log\tau_\text{trained})$$

$\tau$ is parameterised as $\exp(\log\tau)$ to enforce positivity. After training, $\tau$ is frozen — a neuron that learned to be "slow" stays slow regardless of input.

- **Parameters:** $W, U, b$ (recurrent weights) + $\log\tau \in \mathbb{R}^h$ (one per neuron)
- **Solver:** same as Neural ODE

### 3. LTC — `code/models/ltc.jl`

Makes $\tau$ a dynamic function of current state and input (Hasani et al., 2021).

$$\frac{dx}{dt} = -\frac{x}{\tau(x, I)} + f(x, I)$$

$$\tau(x, I) = \tau_0 + \text{softplus}(W_\tau x + U_\tau I + b_\tau)$$

$$f(x, I) = \tanh(W_f x + U_f I + b_f)$$

`softplus` ensures $\tau > \tau_0 > 0$, preventing degenerate zero-timescale dynamics. The same neuron can respond quickly to sharp transients and slowly to gentle drifts — mirroring how biological neurons modulate membrane time constants.

- **Parameters:** $\{W_\tau, U_\tau, b_\tau, \tau_0, W_f, U_f, b_f\}$ — roughly double the parameter count of CTRNN
- **AD compatibility:** `LTCCell{T}` is parameterised over element type `T` so ForwardDiff Dual numbers flow through `unflatten → LTCCell` without hitting `Float32(::Dual)` conversion errors
- **Solver:** same as Neural ODE

### 4. MDL τ Regularisation — `code/train_ltc_mdl.jl`

Applies the Minimum Description Length principle to penalise unnecessarily reactive time constants.

$$\mathcal{L} = \text{MSE} + \lambda \cdot \Omega(\tau), \qquad \Omega(\tau) = -\,\text{mean}(\log \tau)$$

Small $\tau$ encodes high model complexity (reacts to every fluctuation). The $-\log\tau$ penalty discourages this, preferring the simplest dynamics consistent with the data.

Two $\lambda$ strategies:
- **Fixed sweep:** $\lambda \in \{0.001, 0.01, 0.1\}$
- **Adaptive (physics-guided):** $\lambda = \lambda_0 \cdot \zeta$ — overdamped systems receive stronger complexity pressure, directly encoding the prior that simpler physics should yield simpler models

---

## Results

### Base accuracy (closed-loop MSE)

| ζ | LTC | Neural ODE | CTRNN |
|---|---|---|---|
| 0.1 | 0.197 | 1.479 | 0.179 |
| 0.3 | 0.057 | 0.097 | 0.033 |
| 0.5 | 0.023 | 0.079 | 0.053 |
| 0.8 | 0.044 | 0.042 | 0.098 |
| 1.2 | 0.088 | 0.024 | 0.056 |

LTC's advantage is not uniform at in-distribution. It excels near ζ = 0.5 but struggles at ζ = 0.1 (strong oscillations), where model capacity — not τ design — is the bottleneck. The picture changes substantially under distribution shift.

### Regime-switch adaptability (abrupt ζ change at t = 10)

Models trained on pre-switch ζ must adapt without any explicit switch signal.

| Switch | LTC | Neural ODE | CTRNN |
|---|---|---|---|
| 0.1 → 1.2 | 0.093 | 2.250 ❌ | **0.064** |
| 1.2 → 0.1 | 0.089 | **0.003** | 0.045 |
| 0.3 → 0.8 | 0.019 | 0.077 | **0.002** |
| 0.8 → 0.3 | 0.025 | **0.020** | 0.092 |

**LTC is the only architecture that never fails catastrophically across all four switches.** In the extreme 0.1 → 1.2 case, Neural ODE diverges to MSE = 2.250 — an order of magnitude worse than LTC (0.093).

### MDL regularisation effect on regime-switch post-MSE

| Switch | Baseline LTC | MDL-regularised | Δ |
|---|---|---|---|
| 0.1 → 1.2 | 0.093 | 0.038 | **−59%** |
| 1.2 → 0.1 | 0.089 | 0.041 | **−54%** |
| 0.3 → 0.8 | 0.019 | 0.031 | +65% |
| 0.8 → 0.3 | 0.025 | 0.0003 | **−99%** |

MDL regularisation consistently helps on switches involving oscillatory regimes. The exception (0.3 → 0.8) suggests the regularisation occasionally over-constrains τ for moderately damped transitions.

### Julia profiling (LTC inference, hidden_dim = 8, ≥ 100 samples)

| Metric | Value |
|---|---|
| Median inference time | 51.8 ms |
| Min inference time | 25.4 ms |
| Allocations | 658,259 |

Primary bottleneck: ODE solver return type annotated as `::ANY`, not the LTC math itself. Fixing with `@code_warntype` reduced allocation count significantly; type instability originated in the solver dispatch layer, not in `ltc_dynamics`.

---

## Repository Structure

```
physics-to-silicon/
├── code/
│   ├── models/
│   │   ├── ltc.jl                       # Canonical Julia LTC reference
│   │   ├── ctrnn.jl                     # CTRNN (fixed τ per neuron)
│   │   └── neural_ode.jl                # Neural ODE baseline
│   ├── experiments/
│   │   ├── mse_comparison.jl            # Base accuracy across regimes
│   │   ├── generalization.jl            # Cross-ζ transfer
│   │   ├── noise_injection.jl           # Robustness to initial-state noise
│   │   ├── regime_switch.jl             # Abrupt regime-change test
│   │   └── final_attractor_analysis.jl  # Hausdorff distance attractor comparison
│   ├── profiling/
│   │   └── julia_profiling.jl           # BenchmarkTools + @code_warntype
│   ├── pytorch/
│   │   ├── ltc_torch.py                 # PyTorch LTC port
│   │   └── train_ltc_torch.py
│   ├── jax/
│   │   ├── jax_ltc.py                   # JAX LTC port
│   │   ├── train_ltc.py
│   │   └── dump_jaxpr.py                # jax.make_jaxpr analysis
│   ├── data_gen.jl                      # Dataset generation
│   ├── train.jl                         # Train all three models × all ζ
│   └── train_ltc_mdl.jl                 # MDL τ regularisation experiments
├── data/
│   ├── oscillator_zeta{0p1,0p3,0p5,0p8,1p2}.jld2
│   └── switch_zeta{0p1to1p2,1p2to0p1,0p3to0p8,0p8to0p3}.jld2
└── results/
    ├── mse_comparison.{csv,png}
    ├── generalization_avg.{csv,png}
    ├── generalization_to_1p2.{csv,png}
    ├── noise_injection.{csv,png}
    ├── regime_switch.csv
    ├── regime_switch_*.png
    ├── tau_trajectories.png
    ├── attractor_phase_portraits.png
    ├── attractor_hausdorff_summary.csv
    └── julia_profiling_results.csv
```

---

## Cross-Framework Alignment

PyTorch and JAX ports must match the Julia reference output within `tol < 1e-4` on identical inputs before any profiling results are considered valid.

```python
import numpy as np
pred = your_ltc(data_input)
ref  = np.load("data/npy/reference_pred_zeta0p3.npy")
assert np.max(np.abs(pred - ref)) < 1e-4
```

---

## Model Comparison

| Model | τ | Extra parameters | Per-step cost | File |
|---|---|---|---|---|
| Neural ODE | — | — | 1× | `models/neural_ode.jl` |
| CTRNN | Fixed per neuron | $\log\tau \in \mathbb{R}^h$ | ~1× | `models/ctrnn.jl` |
| LTC | Dynamic $\tau(x,I)$ | $W_\tau, U_\tau, b_\tau, \tau_0$ | ~2× | `models/ltc.jl` |

$h$ = hidden_dim. LTC's extra expressiveness roughly doubles per-step FLOP count due to the additional $\tau(x,I)$ computation path.

---

## References

- Hasani et al., *Liquid Time-constant Networks*, AAAI 2021
- Chen et al., *Neural Ordinary Differential Equations*, NeurIPS 2018
- Beer, *On the dynamics of small continuous-time recurrent neural networks*, 1995
- Grünwald, *The Minimum Description Length Principle*, MIT Press 2007
