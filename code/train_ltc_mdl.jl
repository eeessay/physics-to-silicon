# train_ltc_mdl.jl
# Author: Member 1 (Chang)
# EE5311 CA1-21 — LTC training with MDL-inspired τ regularization
#
# Loss = MSE + λ * Ω(τ)
# Ω(τ) = -mean(log τ)   penalises small τ (overly reactive / complex model)
#
# Two λ strategies:
#   1. Fixed λ ∈ {0.001, 0.01, 0.1}  — hyperparameter sweep
#   2. Adaptive λ = λ0 * ζ           — physics-guided: simpler dynamics → stronger regularisation
#
# Only LTC is trained here (Member 1's responsibility).
# Outputs saved to results/ltc_mdl_fixed_lambda{λ}_zeta{ζ}.jld2
#              and results/ltc_mdl_adaptive_zeta{ζ}.jld2

using DifferentialEquations
using SciMLSensitivity
using Zygote
using JLD2
using LinearAlgebra
using Printf
using Statistics
using Random

include("models/ltc.jl")

const ZETAS      = [0.1f0, 0.3f0, 0.5f0, 0.8f0, 1.2f0]
const EPOCHS     = 300
const INPUT_DIM  = 2
const HIDDEN_DIM = 8
const LR         = 1f-3
const GRAD_CLIP  = 1.0f0
const SEED       = 42

# Fixed λ values to sweep
const FIXED_LAMBDAS = [0.001f0, 0.01f0, 0.1f0]
# Adaptive: λ = λ0 * ζ
const LAMBDA0_ADAPTIVE = 0.01f0

Random.seed!(SEED)

# ── Adam ──────────────────────────────────────────────────────────────────────

mutable struct Adam
    lr::Float32; β1::Float32; β2::Float32; ε::Float32
    t::Int; m::Vector{Float32}; v::Vector{Float32}
end
function Adam(n::Int; lr=1f-3, β1=0.9f0, β2=0.999f0, ε=1f-8)
    Adam(lr, β1, β2, ε, 0, zeros(Float32,n), zeros(Float32,n))
end
function adam_step!(opt::Adam, params::Vector{Float32}, grads::Vector{Float32})
    opt.t += 1
    @. opt.m = opt.β1 * opt.m + (1-opt.β1) * grads
    @. opt.v = opt.β2 * opt.v + (1-opt.β2) * grads^2
    m̂ = opt.m ./ (1 - opt.β1^opt.t)
    v̂ = opt.v ./ (1 - opt.β2^opt.t)
    @. params -= opt.lr * m̂ / (sqrt(v̂) + opt.ε)
end
function clip_gradients!(g::Vector{Float32}, max_norm::Float32)
    n = norm(g); n > max_norm && (g .*= max_norm / n)
end

# ── MDL regulariser: Ω(τ) = -mean(log τ) ─────────────────────────────────────
# τ values come from the parameter vector — we compute them analytically
# from p to keep Zygote-differentiable (no ODE needed for τ alone).

function tau_reg(p::AbstractVector, input_dim::Int, hidden_dim::Int)
    # Reconstruct τ₀ from flat param vector (same layout as unflatten)
    h, i = hidden_dim, input_dim
    idx   = h*h + h*i + h   # offset past Wτ, Uτ, bτ
    τ₀    = p[idx+1 : idx+h]
    # τ ≥ τ₀ always (softplus adds ≥ 0), so log(τ₀) is a lower-bound proxy
    # Using τ₀ keeps the regulariser Zygote-friendly without an ODE call
    return -mean(log.(τ₀ .+ 1f-6))
end

# ── AD gradient ───────────────────────────────────────────────────────────────

function ad_gradient(loss_fn::Function, params::Vector{Float32})::Vector{Float32}
    g = Zygote.gradient(loss_fn, params)[1]
    Float32.(something.(g, 0))
end

# ── Training loop ─────────────────────────────────────────────────────────────

function train_ltc_mdl(zeta::Float32, lambda::Float32; label::String="")
    fname = "data/oscillator_zeta$(replace(string(zeta), "." => "p")).jld2"
    data  = load(fname)

    t_train = Float32.(data["t_train"]); x_train = Float32.(data["x_train"])
    t_test  = Float32.(data["t_test"]);  x_test  = Float32.(data["x_test"])

    I_train      = x_train; I_test = x_test
    t_span_train = (t_train[1], t_train[end])
    t_span_test  = (t_test[1],  t_test[end])
    x0           = zeros(Float32, HIDDEN_DIM)

    cell = LTCCell(INPUT_DIM, HIDDEN_DIM)
    p    = flatten(cell)
    opt  = Adam(length(p); lr=LR)

    tag  = isempty(label) ? "" : "_$(label)"
    println("Training LTC+MDL  ζ=$(zeta)  λ=$(lambda)$(isempty(label) ? "" : "  [$label]")")

    best_val = Inf32; best_p = copy(p)

    for epoch in 1:EPOCHS
        function loss_fn(params)
            pred = ltc_rollout_p(params, x0, I_train, t_span_train, t_train,
                                 INPUT_DIM, HIDDEN_DIM)
            out  = pred[:, 1:INPUT_DIM]
            mse  = mse_loss(out, x_train)
            reg  = tau_reg(params, INPUT_DIM, HIDDEN_DIM)
            mse + lambda * reg
        end

        grads = ad_gradient(loss_fn, p)
        clip_gradients!(grads, GRAD_CLIP)
        adam_step!(opt, p, grads)

        if epoch % 30 == 0
            train_loss = loss_fn(p)
            c_val      = unflatten(p, INPUT_DIM, HIDDEN_DIM)
            pred_val   = ltc_rollout(c_val, x0, I_test, t_span_test, t_test)
            val_loss   = mse_loss(pred_val[:, 1:INPUT_DIM], x_test)
            @printf("  epoch %3d/%d  train=%.6f  val=%.6f\n",
                    epoch, EPOCHS, train_loss, val_loss)
            if val_loss < best_val
                best_val = val_loss; best_p = copy(p)
            end
        end
    end

    mkpath("results")
    ckpt = "results/ltc_mdl$(tag)_zeta$(replace(string(zeta), "." => "p")).jld2"
    jldsave(ckpt; params=best_p, val_loss=best_val, lambda=lambda, zeta=zeta)
    println("  Saved → $ckpt  (val_loss=$(round(best_val; digits=6)))\n")
    return unflatten(best_p, INPUT_DIM, HIDDEN_DIM), best_val
end

# ── Main ──────────────────────────────────────────────────────────────────────

function main()
    mkpath("results")

    # 1. Fixed λ sweep
    println("="^60)
    println("MDL regularisation — fixed λ sweep")
    for λ in FIXED_LAMBDAS
        println("\n--- λ = $λ ---")
        for zeta in ZETAS
            tag = "fixed_lambda$(replace(string(λ), "." => "p"))"
            train_ltc_mdl(zeta, λ; label=tag)
        end
    end

    # 2. Adaptive λ = λ0 * ζ
    println("="^60)
    println("MDL regularisation — adaptive λ = $(LAMBDA0_ADAPTIVE) × ζ")
    for zeta in ZETAS
        λ = LAMBDA0_ADAPTIVE * zeta
        train_ltc_mdl(zeta, λ; label="adaptive")
    end

    println("All MDL training complete.")
end

main()
