# experiments/noise_injection.jl
# Author: Member 2 (YAN)
# EE5311 CA1-21 — Noise robustness experiment
# Evaluation mode: autonomous rollout (no teacher forcing)

# Protocol:
#   - Models are trained on clean data only
#   - Clean baseline: seed with clean x0, then autonomous rollout
#   - Noisy test: add noise with multiple σ levels to x0, then autonomous rollout
#   - Target: clean x_test
#   - Metric: MSE(autonomous_pred, clean_target)
#   - Reports degradation ratio = noisy_mse / clean_mse
#   - For each (ζ, σ), repeat noisy-x0 sampling multiple times and average results

using JLD2
using Statistics
using Printf
using Plots
using Random

include("../models/ltc.jl")
include("../models/neural_ode.jl")
include("../models/ctrnn.jl")

const ZETAS  = Float32[0.1f0, 0.3f0, 0.5f0, 0.8f0, 1.2f0]
const INPUT_DIM = 2
const HIDDEN_DIM = 8
const NOISE_LEVELS = Float32[0.01f0, 0.05f0, 0.1f0, 0.2f0]
const BASE_SEED = 42
const N_REPEATS  = 20

mse_loss_common(pred::AbstractMatrix, target::AbstractMatrix) = mean((pred .- target) .^ 2)
zeta_tag(z::Float32) = replace(string(z), "." => "p")

# Load data from .jld2
function load_test_data(zeta::Float32)
    data = load(joinpath(@__DIR__, "..", "..", "data",
                         "oscillator_zeta$(zeta_tag(zeta)).jld2"))
    t_test = Float32.(data["t_test"])
    x_test = Float32.(data["x_test"])
    return t_test, x_test
end

function load_ltc_cell(zeta::Float32)
    p = load(joinpath(@__DIR__, "..", "..", "results",
                      "ltc_zeta$(zeta_tag(zeta)).jld2"), "params")
    return unflatten(p, INPUT_DIM, HIDDEN_DIM)
end
function load_node_cell(zeta::Float32)
    p = load(joinpath(@__DIR__, "..", "..", "results",
                      "node_zeta$(zeta_tag(zeta)).jld2"), "params")
    return unflatten_node(p, INPUT_DIM, HIDDEN_DIM)
end
function load_ctrnn_cell(zeta::Float32)
    p = load(joinpath(@__DIR__, "..", "..", "results",
                      "ctrnn_zeta$(zeta_tag(zeta)).jld2"), "params")
    return unflatten_ctrnn(p, INPUT_DIM, HIDDEN_DIM)
end

# ── Autonomous rollout helpers ────────────────────────────────────────────────
# Two-pass closed-loop evaluation:
#   1. Constant x0 input broadcast over all timesteps (warm-up)
#   2. Shift predicted outputs by one step to use as inputs (closed-loop)

function make_seed_iseq(x0::Vector{Float32}, T::Int)::Matrix{Float32}
    M = Matrix{Float32}(undef, T, length(x0))
    for i in 1:T; M[i, :] = x0; end
    return M
end

function autonomous_rollout_ltc(cell::LTCCell,
                                 x0_seed::Vector{Float32},
                                 t_seq::Vector{Float32})::Matrix{Float32}
    T      = length(t_seq)
    t_span = (t_seq[1], t_seq[end])
    pred   = ltc_rollout(cell, zeros(Float32, HIDDEN_DIM),
                         make_seed_iseq(x0_seed, T), t_span, t_seq)
    out    = pred[:, 1:INPUT_DIM]
    I_seq2 = vcat(reshape(x0_seed, 1, INPUT_DIM), out[1:end-1, :])
    pred2  = ltc_rollout(cell, zeros(Float32, HIDDEN_DIM), I_seq2, t_span, t_seq)
    return pred2[:, 1:INPUT_DIM]
end

function autonomous_rollout_node(cell::NeuralODECell,
                                  x0_seed::Vector{Float32},
                                  t_seq::Vector{Float32})::Matrix{Float32}
    T      = length(t_seq)
    t_span = (t_seq[1], t_seq[end])
    pred   = node_rollout(cell, zeros(Float32, HIDDEN_DIM),
                          make_seed_iseq(x0_seed, T), t_span, t_seq)
    out    = pred[:, 1:INPUT_DIM]
    I_seq2 = vcat(reshape(x0_seed, 1, INPUT_DIM), out[1:end-1, :])
    pred2  = node_rollout(cell, zeros(Float32, HIDDEN_DIM), I_seq2, t_span, t_seq)
    return pred2[:, 1:INPUT_DIM]
end

function autonomous_rollout_ctrnn(cell::CTRNNCell,
                                   x0_seed::Vector{Float32},
                                   t_seq::Vector{Float32})::Matrix{Float32}
    T      = length(t_seq)
    t_span = (t_seq[1], t_seq[end])
    pred   = ctrnn_rollout(cell, zeros(Float32, HIDDEN_DIM),
                           make_seed_iseq(x0_seed, T), t_span, t_seq)
    out    = pred[:, 1:INPUT_DIM]
    I_seq2 = vcat(reshape(x0_seed, 1, INPUT_DIM), out[1:end-1, :])
    pred2  = ctrnn_rollout(cell, zeros(Float32, HIDDEN_DIM), I_seq2, t_span, t_seq)
    return pred2[:, 1:INPUT_DIM]
end

# Noise helper
# Generate one noisy x0 for given (zeta, sigma, repeat_id).
# The same noise seed structure is shared across models for fair comparison.
function make_noisy_x0(x0_clean::Vector{Float32},
                       zeta::Float32,
                       sigma::Float32,
                       repeat_id::Int)::Vector{Float32}
    rng_seed = BASE_SEED +
               round(Int, 1000 * zeta) +
               round(Int, 10000 * sigma) +
               100000 * repeat_id
    rng = MersenneTwister(rng_seed)
    noise = sigma .* randn(rng, Float32, length(x0_clean))
    return x0_clean .+ noise
end


# Evaluation
# For each zeta:
#   - clean baseline: clean x0 → autonomous rollout
#   - noisy test: noisy x0(σ) → autonomous rollout
#   - target is always clean x_test

function eval_ltc_noise(zeta::Float32, sigma::Float32, repeat_id::Int)
    t_test, x_test = load_test_data(zeta)
    cell = load_ltc_cell(zeta)

    x0_clean = vec(x_test[1, :])
    x0_noisy = make_noisy_x0(x0_clean, zeta, sigma, repeat_id)

    clean_pred = autonomous_rollout_ltc(cell, x0_clean, t_test)
    noisy_pred = autonomous_rollout_ltc(cell, x0_noisy, t_test)

    clean_mse = mse_loss_common(clean_pred, x_test)
    noisy_mse = mse_loss_common(noisy_pred, x_test)
    ratio     = noisy_mse / clean_mse

    return clean_mse, noisy_mse, ratio
end

function eval_node_noise(zeta::Float32, sigma::Float32, repeat_id::Int)
    t_test, x_test = load_test_data(zeta)
    cell = load_node_cell(zeta)

    x0_clean = vec(x_test[1, :])
    x0_noisy = make_noisy_x0(x0_clean, zeta, sigma, repeat_id)

    clean_pred = autonomous_rollout_node(cell, x0_clean, t_test)
    noisy_pred = autonomous_rollout_node(cell, x0_noisy, t_test)

    clean_mse = mse_loss_common(clean_pred, x_test)
    noisy_mse = mse_loss_common(noisy_pred, x_test)
    ratio     = noisy_mse / clean_mse

    return clean_mse, noisy_mse, ratio
end

function eval_ctrnn_noise(zeta::Float32, sigma::Float32, repeat_id::Int)
    t_test, x_test = load_test_data(zeta)
    cell = load_ctrnn_cell(zeta)

    x0_clean = vec(x_test[1, :])
    x0_noisy = make_noisy_x0(x0_clean, zeta, sigma, repeat_id)

    clean_pred = autonomous_rollout_ctrnn(cell, x0_clean, t_test)
    noisy_pred = autonomous_rollout_ctrnn(cell, x0_noisy, t_test)

    clean_mse = mse_loss_common(clean_pred, x_test)
    noisy_mse = mse_loss_common(noisy_pred, x_test)
    ratio     = noisy_mse / clean_mse

    return clean_mse, noisy_mse, ratio
end


function main()
    results_dir = joinpath(@__DIR__, "..", "..", "results")
    mkpath(results_dir)

    n_levels = length(NOISE_LEVELS)
    n_zeta   = length(ZETAS)

    ltc_ratio   = zeros(Float32, n_levels, n_zeta)
    node_ratio  = zeros(Float32, n_levels, n_zeta)
    ctrnn_ratio = zeros(Float32, n_levels, n_zeta)

    open(joinpath(results_dir, "noise_injection.csv"), "w") do io
        println(io, "zeta,noise_std,ltc_clean,ltc_noisy_mean,ltc_ratio_mean,node_clean,node_noisy_mean,node_ratio_mean,ctrnn_clean,ctrnn_noisy_mean,ctrnn_ratio_mean")

        for (j, z) in enumerate(ZETAS)
            for (i, sigma) in enumerate(NOISE_LEVELS)
                ltc_clean_vals   = Float32[]
                ltc_noisy_vals   = Float32[]
                ltc_ratio_vals   = Float32[]

                node_clean_vals  = Float32[]
                node_noisy_vals  = Float32[]
                node_ratio_vals  = Float32[]

                ctrnn_clean_vals = Float32[]
                ctrnn_noisy_vals = Float32[]
                ctrnn_ratio_vals = Float32[]

                for rep in 1:N_REPEATS
                    lc, ln, lr = eval_ltc_noise(z, sigma, rep)
                    nc, nn, nr = eval_node_noise(z, sigma, rep)
                    cc, cn, cr = eval_ctrnn_noise(z, sigma, rep)

                    push!(ltc_clean_vals, lc)
                    push!(ltc_noisy_vals, ln)
                    push!(ltc_ratio_vals, lr)

                    push!(node_clean_vals, nc)
                    push!(node_noisy_vals, nn)
                    push!(node_ratio_vals, nr)

                    push!(ctrnn_clean_vals, cc)
                    push!(ctrnn_noisy_vals, cn)
                    push!(ctrnn_ratio_vals, cr)
                end

                ltc_clean_mean   = mean(ltc_clean_vals)
                ltc_noisy_mean   = mean(ltc_noisy_vals)
                ltc_ratio_mean   = mean(ltc_ratio_vals)

                node_clean_mean  = mean(node_clean_vals)
                node_noisy_mean  = mean(node_noisy_vals)
                node_ratio_mean  = mean(node_ratio_vals)

                ctrnn_clean_mean = mean(ctrnn_clean_vals)
                ctrnn_noisy_mean = mean(ctrnn_noisy_vals)
                ctrnn_ratio_mean = mean(ctrnn_ratio_vals)

                ltc_ratio[i, j]   = ltc_ratio_mean
                node_ratio[i, j]  = node_ratio_mean
                ctrnn_ratio[i, j] = ctrnn_ratio_mean

                @printf(io, "%.1f,%.2f,%.6e,%.6e,%.4f,%.6e,%.6e,%.4f,%.6e,%.6e,%.4f\n",
                        z, sigma,
                        ltc_clean_mean, ltc_noisy_mean, ltc_ratio_mean,
                        node_clean_mean, node_noisy_mean, node_ratio_mean,
                        ctrnn_clean_mean, ctrnn_noisy_mean, ctrnn_ratio_mean)

                @printf("ζ=%.1f  σ=%.2f  LTC=%.3f  NODE=%.3f  CTRNN=%.3f\n",
                        z, sigma, ltc_ratio_mean, node_ratio_mean, ctrnn_ratio_mean)
            end
        end
    end

    # Average degradation ratio vs sigma
    ltc_avg   = vec(mean(ltc_ratio, dims=2))
    node_avg  = vec(mean(node_ratio, dims=2))
    ctrnn_avg = vec(mean(ctrnn_ratio, dims=2))

    p = plot(NOISE_LEVELS, ltc_avg;
             label="LTC", marker=:circle, linewidth=2,
             xlabel="Initial-state noise std σ",
             ylabel="Mean degradation ratio (noisy/clean MSE)",
             title="Noise robustness — autonomous rollout",
             legend=:topleft)
    plot!(p, NOISE_LEVELS, node_avg;  label="Neural ODE", marker=:square,  linewidth=2)
    plot!(p, NOISE_LEVELS, ctrnn_avg; label="CTRNN",      marker=:diamond, linewidth=2)
    hline!(p, [1.0]; linestyle=:dash, label="baseline")

    savefig(p, joinpath(results_dir, "noise_injection.png"))

    # Per-zeta degradation ratio at sigma = 0.1
    idx_01 = findfirst(==(0.1f0), NOISE_LEVELS)
    p2 = plot(ZETAS, ltc_ratio[idx_01, :];
              label="LTC", marker=:circle, linewidth=2,
              xlabel="ζ", ylabel="Degradation ratio at σ = 0.1",
              title="Noise robustness per ζ — autonomous rollout",
              legend=:topright)
    plot!(p2, ZETAS, node_ratio[idx_01, :];  label="Neural ODE", marker=:square,  linewidth=2)
    plot!(p2, ZETAS, ctrnn_ratio[idx_01, :]; label="CTRNN",      marker=:diamond, linewidth=2)
    hline!(p2, [1.0]; linestyle=:dash, label="baseline")

    savefig(p2, joinpath(results_dir, "noise_injection_per_zeta.png"))

    println("\nSaved results/noise_injection.csv")
    println("Saved results/noise_injection.png")
    println("Saved results/noise_injection_per_zeta.png")
end

main()