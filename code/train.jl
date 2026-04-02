# train.jl
# Author: Member 1 (Chang)
# EE5311 CA1-21 — LTC/NeuralODE/CTRNN training script
#
# v2 (AD): Gradients computed via Zygote.gradient + SciMLSensitivity
#          (ForwardDiffSensitivity).  The fd_gradient fallback is removed.
#
# Trains all models on all ζ values, saves:
#   results/ltc_zeta<ζ>.jld2
#   results/node_zeta<ζ>.jld2
#   results/ctrnn_zeta<ζ>.jld2
#   results/julia_ltc_reference_output.jld2   (alignment artefact for Members 3/4/5)

using DifferentialEquations
using SciMLSensitivity
using Zygote
using JLD2
using LinearAlgebra
using Printf
using Statistics
using Random

include("models/ltc.jl")
include("models/neural_ode.jl")
include("models/ctrnn.jl")

mse_loss_common(pred::AbstractMatrix, target::AbstractMatrix) =
    mean((pred .- target) .^ 2)

# ── Hyperparameters ───────────────────────────────────────────────────────────

const ZETAS      = [0.1f0, 0.3f0, 0.5f0, 0.8f0, 1.2f0]
const EPOCHS     = 300
const INPUT_DIM  = 2
const HIDDEN_DIM = 8
const LR         = 1f-3
const GRAD_CLIP  = 1.0f0
const SEED       = 42

Random.seed!(SEED)

# ── Minimal Adam optimizer ────────────────────────────────────────────────────

mutable struct Adam
    lr::Float32
    β1::Float32
    β2::Float32
    ε::Float32
    t::Int
    m::Vector{Float32}
    v::Vector{Float32}
end

function Adam(n_params::Int; lr=1f-3, β1=0.9f0, β2=0.999f0, ε=1f-8)
    Adam(lr, β1, β2, ε, 0, zeros(Float32, n_params), zeros(Float32, n_params))
end

function adam_step!(opt::Adam, params::Vector{Float32}, grads::Vector{Float32})
    opt.t += 1
    @. opt.m = opt.β1 * opt.m + (1 - opt.β1) * grads
    @. opt.v = opt.β2 * opt.v + (1 - opt.β2) * grads^2
    m̂ = opt.m ./ (1 - opt.β1^opt.t)
    v̂ = opt.v ./ (1 - opt.β2^opt.t)
    @. params -= opt.lr * m̂ / (sqrt(v̂) + opt.ε)
end

function clip_gradients!(g::Vector{Float32}, max_norm::Float32)
    n = norm(g)
    if n > max_norm
        g .*= max_norm / n
    end
end

# ── AD gradient helper ────────────────────────────────────────────────────────
# Zygote.gradient returns a 1-tuple; we extract and convert to Float32.

function ad_gradient(loss_fn::Function, params::Vector{Float32})::Vector{Float32}
    g = Zygote.gradient(loss_fn, params)[1]
    g === nothing && return zeros(Float32, length(params))
    return Float32.(something.(g, 0))
end

# ── Training loop: LTC ───────────────────────────────────────────────────────

function train_ltc(zeta::Float32)
    fname = "data/oscillator_zeta$(replace(string(zeta), "." => "p")).jld2"
    data  = load(fname)

    t_train = Float32.(data["t_train"])
    x_train = Float32.(data["x_train"])
    t_test  = Float32.(data["t_test"])
    x_test  = Float32.(data["x_test"])

    I_train = x_train
    I_test  = x_test

    t_span_train = (t_train[1], t_train[end])
    t_span_test  = (t_test[1],  t_test[end])

    x0 = zeros(Float32, HIDDEN_DIM)

    cell = LTCCell(INPUT_DIM, HIDDEN_DIM)
    p    = flatten(cell)
    n_p  = length(p)
    opt  = Adam(n_p; lr=LR)

    println("Training LTC on ζ = $zeta  ($n_p parameters)  [AD]")

    best_val_loss = Inf32
    best_p        = copy(p)

    for epoch in 1:EPOCHS
        # Loss as a function of the flat param vector — Zygote traces through this
        function loss_fn(params)
            pred = ltc_rollout_p(params, x0, I_train, t_span_train, t_train,
                                 INPUT_DIM, HIDDEN_DIM)
            out  = pred[:, 1:INPUT_DIM]
            mse_loss(out, x_train)
        end

        grads = ad_gradient(loss_fn, p)
        clip_gradients!(grads, GRAD_CLIP)
        adam_step!(opt, p, grads)

        if epoch % 10 == 0
            train_loss = loss_fn(p)

            c_val    = unflatten(p, INPUT_DIM, HIDDEN_DIM)
            pred_val = ltc_rollout(c_val, x0, I_test, t_span_test, t_test)
            val_loss = mse_loss(pred_val[:, 1:INPUT_DIM], x_test)

            @printf("  ζ=%.1f  epoch %3d/%d  train=%.6f  val=%.6f\n",
                    zeta, epoch, EPOCHS, train_loss, val_loss)

            if val_loss < best_val_loss
                best_val_loss = val_loss
                best_p        = copy(p)
            end
        end
    end

    mkpath("results")
    best_cell  = unflatten(best_p, INPUT_DIM, HIDDEN_DIM)
    ckpt_path  = "results/ltc_zeta$(replace(string(zeta), "." => "p")).jld2"
    jldsave(ckpt_path; params=best_p, val_loss=best_val_loss)
    println("  Saved → $ckpt_path  (val_loss=$(round(best_val_loss; digits=6)))")

    return best_cell, best_val_loss
end

# ── Training loop: Neural ODE ────────────────────────────────────────────────

function train_neural_ode(zeta::Float32)
    fname = "data/oscillator_zeta$(replace(string(zeta), "." => "p")).jld2"
    data  = load(fname)

    t_train = Float32.(data["t_train"])
    x_train = Float32.(data["x_train"])
    t_test  = Float32.(data["t_test"])
    x_test  = Float32.(data["x_test"])

    I_train      = x_train
    I_test       = x_test
    t_span_train = (t_train[1], t_train[end])
    t_span_test  = (t_test[1],  t_test[end])
    x0           = zeros(Float32, HIDDEN_DIM)

    cell = NeuralODECell(INPUT_DIM, HIDDEN_DIM)
    p    = flatten_node(cell)
    n_p  = length(p)
    opt  = Adam(n_p; lr=LR)

    println("Training Neural ODE on ζ = $zeta  ($n_p parameters)  [AD]")

    best_val_loss = Inf32
    best_p        = copy(p)

    for epoch in 1:EPOCHS
        function loss_fn(params)
            pred = node_rollout_p(params, x0, I_train, t_span_train, t_train,
                                  INPUT_DIM, HIDDEN_DIM)
            out  = pred[:, 1:INPUT_DIM]
            mse_loss_common(out, x_train)
        end

        grads = ad_gradient(loss_fn, p)
        clip_gradients!(grads, GRAD_CLIP)
        adam_step!(opt, p, grads)

        if epoch % 10 == 0
            train_loss = loss_fn(p)

            c_val    = unflatten_node(p, INPUT_DIM, HIDDEN_DIM)
            pred_val = node_rollout(c_val, x0, I_test, t_span_test, t_test)
            val_loss = mse_loss_common(pred_val[:, 1:INPUT_DIM], x_test)

            @printf("  [NeuralODE] ζ=%.1f  epoch %3d/%d  train=%.6f  val=%.6f\n",
                    zeta, epoch, EPOCHS, train_loss, val_loss)

            if val_loss < best_val_loss
                best_val_loss = val_loss
                best_p        = copy(p)
            end
        end
    end

    mkpath("results")
    ckpt_path = "results/node_zeta$(replace(string(zeta), "." => "p")).jld2"
    jldsave(ckpt_path; params=best_p, val_loss=best_val_loss)
    println("  Saved → $ckpt_path  (val_loss=$(round(best_val_loss; digits=6)))")

    return unflatten_node(best_p, INPUT_DIM, HIDDEN_DIM), best_val_loss
end

# ── Training loop: CTRNN ─────────────────────────────────────────────────────

function train_ctrnn(zeta::Float32)
    fname = "data/oscillator_zeta$(replace(string(zeta), "." => "p")).jld2"
    data  = load(fname)

    t_train = Float32.(data["t_train"])
    x_train = Float32.(data["x_train"])
    t_test  = Float32.(data["t_test"])
    x_test  = Float32.(data["x_test"])

    I_train      = x_train
    I_test       = x_test
    t_span_train = (t_train[1], t_train[end])
    t_span_test  = (t_test[1],  t_test[end])
    x0           = zeros(Float32, HIDDEN_DIM)

    cell = CTRNNCell(INPUT_DIM, HIDDEN_DIM)
    p    = flatten_ctrnn(cell)
    n_p  = length(p)
    opt  = Adam(n_p; lr=LR)

    println("Training CTRNN on ζ = $zeta  ($n_p parameters)  [AD]")

    best_val_loss = Inf32
    best_p        = copy(p)

    for epoch in 1:EPOCHS
        function loss_fn(params)
            pred = ctrnn_rollout_p(params, x0, I_train, t_span_train, t_train,
                                   INPUT_DIM, HIDDEN_DIM)
            out  = pred[:, 1:INPUT_DIM]
            mse_loss_common(out, x_train)
        end

        grads = ad_gradient(loss_fn, p)
        clip_gradients!(grads, GRAD_CLIP)
        adam_step!(opt, p, grads)

        if epoch % 10 == 0
            train_loss = loss_fn(p)

            c_val    = unflatten_ctrnn(p, INPUT_DIM, HIDDEN_DIM)
            pred_val = ctrnn_rollout(c_val, x0, I_test, t_span_test, t_test)
            val_loss = mse_loss_common(pred_val[:, 1:INPUT_DIM], x_test)

            @printf("  [CTRNN] ζ=%.1f  epoch %3d/%d  train=%.6f  val=%.6f\n",
                    zeta, epoch, EPOCHS, train_loss, val_loss)

            if val_loss < best_val_loss
                best_val_loss = val_loss
                best_p        = copy(p)
            end
        end
    end

    mkpath("results")
    ckpt_path = "results/ctrnn_zeta$(replace(string(zeta), "." => "p")).jld2"
    jldsave(ckpt_path; params=best_p, val_loss=best_val_loss)
    println("  Saved → $ckpt_path  (val_loss=$(round(best_val_loss; digits=6)))")

    return unflatten_ctrnn(best_p, INPUT_DIM, HIDDEN_DIM), best_val_loss
end

# ── Main ──────────────────────────────────────────────────────────────────────

function main()
    mkpath("results")

    results_ltc   = Dict{Float32,Float32}()
    results_node  = Dict{Float32,Float32}()
    results_ctrnn = Dict{Float32,Float32}()

    println("="^60)
    println("Training LTC  [Zygote AD + ForwardDiffSensitivity]")
    for zeta in ZETAS
        _, val_loss = train_ltc(zeta)
        results_ltc[zeta] = val_loss
        println()
    end

    println("="^60)
    println("Training Neural ODE  [Zygote AD + ForwardDiffSensitivity]")
    for zeta in ZETAS
        _, val_loss = train_neural_ode(zeta)
        results_node[zeta] = val_loss
        println()
    end

    println("="^60)
    println("Training CTRNN  [Zygote AD + ForwardDiffSensitivity]")
    for zeta in ZETAS
        _, val_loss = train_ctrnn(zeta)
        results_ctrnn[zeta] = val_loss
        println()
    end

    println("="^70)
    println("Training complete. Summary:")
    @printf("%-8s  %-12s  %-12s  %-12s\n", "ζ", "LTC", "NeuralODE", "CTRNN")
    for zeta in ZETAS
        @printf("%-8.1f  %-12.6e  %-12.6e  %-12.6e\n",
                zeta,
                results_ltc[zeta],
                results_node[zeta],
                results_ctrnn[zeta])
    end

    # ── Reference output for alignment (ζ=0.3, LTC) ─────────────────────────
    ref_zeta  = 0.3f0
    ref_fname = "data/oscillator_zeta0p3.jld2"
    ref_data  = load(ref_fname)
    ref_p     = load("results/ltc_zeta0p3.jld2", "params")
    ref_cell  = unflatten(ref_p, INPUT_DIM, HIDDEN_DIM)
    t_test    = Float32.(ref_data["t_test"])
    x_test    = Float32.(ref_data["x_test"])
    x0        = zeros(Float32, HIDDEN_DIM)
    pred      = ltc_rollout(ref_cell, x0, x_test,
                             (t_test[1], t_test[end]), t_test)
    ref_out   = pred[:, 1:INPUT_DIM]

    jldsave("results/julia_ltc_reference_output.jld2";
            pred=ref_out, zeta=ref_zeta, t=t_test)
    println("\nReference output saved → results/julia_ltc_reference_output.jld2")
    println("Members 3/4/5: use this file for alignment verification (tol < 1e-4)")
end

main()
