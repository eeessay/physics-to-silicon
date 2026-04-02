# experiments/mse_comparison.jl
# Author: Member 2 (YAN)
# EE5311 CA1-21 — Seen ζ MSE comparison

# Evaluation mode: autonomous rollout
#   - Warm-up with x0 broadcast over all timesteps
#   - Uses its own predicted outputs (first 2 dims) as inputs
#   - No teacher forcing: evaluates closed-loop predictive dynamics

using JLD2
using Statistics
using Printf
using Plots

include("../models/ltc.jl")
include("../models/neural_ode.jl")
include("../models/ctrnn.jl")

const ZETAS = Float32[0.1f0, 0.3f0, 0.5f0, 0.8f0, 1.2f0]
const INPUT_DIM = 2
const HIDDEN_DIM = 8

mse_loss_common(pred::Matrix{Float32}, target::Matrix{Float32}) = mean((pred .- target) .^ 2)
zeta_tag(z::Float32) = replace(string(z), "." => "p")

function load_test_data(zeta::Float32)
    data_path = joinpath(@__DIR__, "..", "..", "data", "oscillator_zeta$(zeta_tag(zeta)).jld2")
    data = load(data_path)
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
    # Explicitly construct (T, INPUT_DIM) Matrix — avoids Adjoint type issues
    M = Matrix{Float32}(undef, T, length(x0))
    for i in 1:T; M[i, :] = x0; end
    return M
end


# Autonomous rollout

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

# Evaluation
function eval_ltc(zeta::Float32)
    t_test, x_test = load_test_data(zeta)
    cell = load_ltc_cell(zeta)
    out  = autonomous_rollout_ltc(cell, x_test[1, :], t_test)
    return mse_loss_common(out, x_test)
end

function eval_node(zeta::Float32)
    t_test, x_test = load_test_data(zeta)
    cell = load_node_cell(zeta)
    out  = autonomous_rollout_node(cell, x_test[1, :], t_test)
    return mse_loss_common(out, x_test)
end

function eval_ctrnn(zeta::Float32)
    t_test, x_test = load_test_data(zeta)
    cell = load_ctrnn_cell(zeta)
    out  = autonomous_rollout_ctrnn(cell, x_test[1, :], t_test)
    return mse_loss_common(out, x_test)
end

function main()
    results_dir = joinpath(@__DIR__, "..", "..", "results")
    mkpath(results_dir)

    ltc_vals   = Float32[]
    node_vals  = Float32[]
    ctrnn_vals = Float32[]

    open(joinpath(results_dir, "mse_comparison.csv"), "w") do io
        println(io, "zeta,ltc_mse,node_mse,ctrnn_mse")
        for z in ZETAS
            ltc_loss  = eval_ltc(z)
            node_loss = eval_node(z)
            ctr_loss  = eval_ctrnn(z)
            push!(ltc_vals,   ltc_loss)
            push!(node_vals,  node_loss)
            push!(ctrnn_vals, ctr_loss)
            @printf(io, "%.1f,%.8e,%.8e,%.8e\n", z, ltc_loss, node_loss, ctr_loss)
            @printf("ζ=%.1f  LTC=%.6f  NODE=%.6f  CTRNN=%.6f\n",
                    z, ltc_loss, node_loss, ctr_loss)
        end
    end

    p = plot(ZETAS, ltc_vals;
             label="LTC", marker=:circle, linewidth=2,
             xlabel="ζ", ylabel="Test MSE (autonomous rollout)",
             yscale=:log10, title="MSE comparison — autonomous rollout")
    plot!(p, ZETAS, node_vals;  label="Neural ODE", marker=:square,  linewidth=2)
    plot!(p, ZETAS, ctrnn_vals; label="CTRNN",      marker=:diamond, linewidth=2)
    savefig(p, joinpath(results_dir, "mse_comparison.png"))

    println("\nSaved results/mse_comparison.csv")
    println("Saved results/mse_comparison.png")
end

main()
