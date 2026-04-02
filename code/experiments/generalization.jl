# experiments/generalization.jl
# Author: Member 2 (YAN)
# EE5311 CA1-21 — Cross-ζ generalization test

# Evaluation mode: autonomous rollout (no teacher forcing)
# Data source: .jld2 files

# Two experiments:
#   1. Full cross-ζ: train on each ζ, test on all other ζ
#   2. Train on ζ∈{0.1,0.3,0.5,0.8}, test on ζ=1.2

using Statistics
using Printf
using Plots
using JLD2

include("../models/ltc.jl")
include("../models/neural_ode.jl")
include("../models/ctrnn.jl")

const ZETAS     = Float32[0.1f0, 0.3f0, 0.5f0, 0.8f0, 1.2f0]
const INPUT_DIM = 2
const HIDDEN_DIM = 8

mse_loss_common(pred::Matrix{Float32}, target::Matrix{Float32}) = mean((pred .- target) .^ 2)
zeta_tag(z::Float32) = replace(string(z), "." => "p")

# Load data from .jld2 (interleaved split)

function load_test_data(zeta::Float32)
    data_path = joinpath(@__DIR__, "..", "..", "data", "oscillator_zeta$(zeta_tag(zeta)).jld2")
    data  = load(data_path)
    t_test = Float32.(data["t_test"])
    x_test = Float32.(data["x_test"])
    return t_test, x_test
end

# Load pre-trained model checkpoints
function load_ltc_cell(zeta::Float32)
    p = load(joinpath(@__DIR__, "..", "..", "results", "ltc_zeta$(zeta_tag(zeta)).jld2"), "params")
    return unflatten(p, INPUT_DIM, HIDDEN_DIM)
end
function load_node_cell(zeta::Float32)
    p = load(joinpath(@__DIR__, "..", "..", "results", "node_zeta$(zeta_tag(zeta)).jld2"), "params")
    return unflatten_node(p, INPUT_DIM, HIDDEN_DIM)
end
function load_ctrnn_cell(zeta::Float32)
    p = load(joinpath(@__DIR__, "..", "..", "results", "ctrnn_zeta$(zeta_tag(zeta)).jld2"), "params")
    return unflatten_ctrnn(p, INPUT_DIM, HIDDEN_DIM)
end

# ── Autonomous rollout helpers ────────────────────────────────────────────────
# Two-pass closed-loop evaluation:
#  1. Constant x0 input broadcast over all timesteps (warm-up)
#  2. Shift predicted outputs by one step to use as inputs (closed-loop)

function make_seed_iseq(x0::Vector{Float32}, T::Int)::Matrix{Float32}
    M = Matrix{Float32}(undef, T, length(x0))
    for i in 1:T; M[i, :] = x0; end
    return M
end

# Autonomous trajectory rollout
function autonomous_rollout_ltc(cell::LTCCell,
                                 x0_obs::Vector{Float32},
                                 t_seq::Vector{Float32})::Matrix{Float32}
    T      = length(t_seq)
    t_span = (t_seq[1], t_seq[end])
    pred   = ltc_rollout(cell, zeros(Float32, HIDDEN_DIM),
                         make_seed_iseq(x0_obs, T), t_span, t_seq)
    out    = pred[:, 1:INPUT_DIM]
    I_seq2 = vcat(reshape(x0_obs, 1, INPUT_DIM), out[1:end-1, :])
    pred2  = ltc_rollout(cell, zeros(Float32, HIDDEN_DIM), I_seq2, t_span, t_seq)
    return pred2[:, 1:INPUT_DIM]
end

function autonomous_rollout_node(cell::NeuralODECell,
                                  x0_obs::Vector{Float32},
                                  t_seq::Vector{Float32})::Matrix{Float32}
    T      = length(t_seq)
    t_span = (t_seq[1], t_seq[end])
    pred   = node_rollout(cell, zeros(Float32, HIDDEN_DIM),
                          make_seed_iseq(x0_obs, T), t_span, t_seq)
    out    = pred[:, 1:INPUT_DIM]
    I_seq2 = vcat(reshape(x0_obs, 1, INPUT_DIM), out[1:end-1, :])
    pred2  = node_rollout(cell, zeros(Float32, HIDDEN_DIM), I_seq2, t_span, t_seq)
    return pred2[:, 1:INPUT_DIM]
end

function autonomous_rollout_ctrnn(cell::CTRNNCell,
                                   x0_obs::Vector{Float32},
                                   t_seq::Vector{Float32})::Matrix{Float32}
    T      = length(t_seq)
    t_span = (t_seq[1], t_seq[end])
    pred   = ctrnn_rollout(cell, zeros(Float32, HIDDEN_DIM),
                           make_seed_iseq(x0_obs, T), t_span, t_seq)
    out    = pred[:, 1:INPUT_DIM]
    I_seq2 = vcat(reshape(x0_obs, 1, INPUT_DIM), out[1:end-1, :])
    pred2  = ctrnn_rollout(cell, zeros(Float32, HIDDEN_DIM), I_seq2, t_span, t_seq)
    return pred2[:, 1:INPUT_DIM]
end

# Transfer evaluation: train on one ζ, test on another

function eval_ltc_transfer(train_zeta::Float32, test_zeta::Float32)
    t_test, x_test = load_test_data(test_zeta)
    cell = load_ltc_cell(train_zeta)
    out  = autonomous_rollout_ltc(cell, x_test[1, :], t_test)
    return mse_loss_common(out, x_test)
end

function eval_node_transfer(train_zeta::Float32, test_zeta::Float32)
    t_test, x_test = load_test_data(test_zeta)
    cell = load_node_cell(train_zeta)
    out  = autonomous_rollout_node(cell, x_test[1, :], t_test)
    return mse_loss_common(out, x_test)
end

function eval_ctrnn_transfer(train_zeta::Float32, test_zeta::Float32)
    t_test, x_test = load_test_data(test_zeta)
    cell = load_ctrnn_cell(train_zeta)
    out  = autonomous_rollout_ctrnn(cell, x_test[1, :], t_test)
    return mse_loss_common(out, x_test)
end

function mean_by_test(mat::Matrix{Float32})
    [mean(filter(!isnan, [mat[i,j] for i in 1:size(mat,1) if i != j]))
     for j in 1:size(mat,2)]
end

function main()
    results_dir = joinpath(@__DIR__, "..", "..", "results")
    mkpath(results_dir)
    n = length(ZETAS)

    ltc_mat   = fill(NaN32, n, n)
    node_mat  = fill(NaN32, n, n)
    ctrnn_mat = fill(NaN32, n, n)

    println("Running cross-ζ transfer evaluation (autonomous rollout)...")
    for (i, train_z) in enumerate(ZETAS)
        for (j, test_z) in enumerate(ZETAS)
            train_z == test_z && continue
            ltc_mat[i,j]   = eval_ltc_transfer(train_z, test_z)
            node_mat[i,j]  = eval_node_transfer(train_z, test_z)
            ctrnn_mat[i,j] = eval_ctrnn_transfer(train_z, test_z)
            @printf("  train=%.1f → test=%.1f  LTC=%.5f  NODE=%.5f  CTRNN=%.5f\n",
                    train_z, test_z,
                    ltc_mat[i,j], node_mat[i,j], ctrnn_mat[i,j])
        end
    end

    ltc_by_test   = mean_by_test(ltc_mat)
    node_by_test  = mean_by_test(node_mat)
    ctrnn_by_test = mean_by_test(ctrnn_mat)

    open(joinpath(results_dir, "generalization_avg.csv"), "w") do io
        println(io, "test_zeta,ltc_mean_mse,node_mean_mse,ctrnn_mean_mse")
        for k in eachindex(ZETAS)
            @printf(io, "%.1f,%.8e,%.8e,%.8e\n",
                    ZETAS[k], ltc_by_test[k], node_by_test[k], ctrnn_by_test[k])
        end
    end

    p = plot(ZETAS, ltc_by_test;
             label="LTC", marker=:circle, linewidth=2,
             xlabel="Test ζ", ylabel="Mean transfer MSE (autonomous)",
             yscale=:log10, title="Generalization by test ζ — autonomous rollout")
    plot!(p, ZETAS, node_by_test;  label="Neural ODE", marker=:square,  linewidth=2)
    plot!(p, ZETAS, ctrnn_by_test; label="CTRNN",      marker=:diamond, linewidth=2)
    savefig(p, joinpath(results_dir, "generalization_avg.png"))

    println("\nSaved results/generalization_avg.csv / .png")

    # Train on {0.1,0.3,0.5,0.8}, test on 1.2
    TARGET = 1.2f0
    TRAIN_ZETAS = Float32[0.1f0, 0.3f0, 0.5f0, 0.8f0]

    ltc_to_12   = Float32[]
    node_to_12  = Float32[]
    ctrnn_to_12 = Float32[]

    println("\nGeneralization to ζ=1.2:")
    open(joinpath(results_dir, "generalization_to_1p2.csv"), "w") do io
        println(io, "train_zeta,test_zeta,ltc_mse,node_mse,ctrnn_mse")
        for train_z in TRAIN_ZETAS
            l = eval_ltc_transfer(train_z, TARGET)
            nd = eval_node_transfer(train_z, TARGET)
            c = eval_ctrnn_transfer(train_z, TARGET)
            push!(ltc_to_12, l); push!(node_to_12, nd); push!(ctrnn_to_12, c)
            @printf(io, "%.1f,%.1f,%.8e,%.8e,%.8e\n", train_z, TARGET, l, nd, c)
            @printf("  train=%.1f  LTC=%.5f  NODE=%.5f  CTRNN=%.5f\n",
                    train_z, l, nd, c)
        end
    end

    p2 = plot(TRAIN_ZETAS, ltc_to_12;
              label="LTC", marker=:circle, linewidth=2,
              xlabel="Train ζ", ylabel="Transfer MSE on ζ=1.2 (autonomous)",
              yscale=:log10, title="Generalization to ζ=1.2 — autonomous rollout")
    plot!(p2, TRAIN_ZETAS, node_to_12;  label="Neural ODE", marker=:square,  linewidth=2)
    plot!(p2, TRAIN_ZETAS, ctrnn_to_12; label="CTRNN",      marker=:diamond, linewidth=2)
    savefig(p2, joinpath(results_dir, "generalization_to_1p2.png"))

    println("\nSaved results/generalization_to_1p2.csv / .png")
end

main()
