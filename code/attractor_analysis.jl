# attractor_analysis.jl
# Author: Member 5
# EE5311 CA1-21 — Attractor analysis
# From 100 random initial conditions, compare trajectory convergence structure
# across: true system, Neural ODE, CTRNN, LTC

using DifferentialEquations
using JLD2

include("code/models/ltc.jl")
include("code/models/ctrnn.jl")
include("code/models/neural_ode.jl")

const N_ICS     = 100
const IC_RANGE  = (-2.0f0, 2.0f0)
const ZETA_REF  = 0.3f0   # reference damping regime for attractor analysis

# ── Hausdorff distance ────────────────────────────────────────────────────────

function hausdorff_distance(A::Matrix{Float32}, B::Matrix{Float32})::Float32
    d_AB = maximum(minimum(sqrt.(sum((A[i:i,:] .- B).^2; dims=2))) for i in axes(A,1))
    d_BA = maximum(minimum(sqrt.(sum((B[i:i,:] .- A).^2; dims=2))) for i in axes(B,1))
    max(d_AB, d_BA)
end

# ── TODO (Member 5) ───────────────────────────────────────────────────────────
# 1. Sample N_ICS initial conditions uniformly in IC_RANGE²
# 2. Integrate true damped oscillator from each IC
# 3. Integrate each trained model from each IC
# 4. Compute Hausdorff distance: model attractor vs true attractor
# 5. Plot phase portraits; save to results/attractor_*.png
# 6. Print summary table of Hausdorff distances

println("Attractor analysis — TODO: implement after models are trained")
