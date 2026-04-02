# experiments/regime_switch.jl
# Author: Member 1 (Chang)
# EE5311 CA1-21 — Regime-switching experiment
#
# Evaluates all three models on trajectories where ζ switches at t=10.
# Models were trained on single-ζ data — they must adapt to the switch
# without any explicit signal.
#
# Metrics:
#   - Overall MSE on the test split
#   - Pre-switch MSE  (t < T_SWITCH)
#   - Post-switch MSE (t ≥ T_SWITCH)
#   - Recovery steps: how many test steps after switch until MSE < threshold

using JLD2
using Statistics
using Printf
using Plots

include("../models/ltc.jl")
include("../models/neural_ode.jl")
include("../models/ctrnn.jl")

const INPUT_DIM  = 2
const HIDDEN_DIM = 8
const T_SWITCH   = 10.0f0
const RECOVERY_THRESHOLD = 0.01f0   # MSE per-step threshold for "recovered"

mse_loss_common(a, b) = mean((a .- b).^2)
zeta_tag(z::Float32)  = replace(string(z), "." => "p")

const SWITCH_PAIRS = [
    (0.1f0, 1.2f0),
    (1.2f0, 0.1f0),
    (0.3f0, 0.8f0),
    (0.8f0, 0.3f0),
]

# ── Data and model loading ────────────────────────────────────────────────────

function load_switch_data(z1::Float32, z2::Float32)
    path = joinpath(@__DIR__, "..", "..", "data",
                    "switch_zeta$(zeta_tag(z1))to$(zeta_tag(z2)).jld2")
    d = load(path)
    return Float32.(d["t_test"]), Float32.(d["x_test"])
end

function load_ltc_cell(zeta::Float32; mdl_label::String="")
    tag  = isempty(mdl_label) ? "" : "_$(mdl_label)"
    path = joinpath(@__DIR__, "..", "..", "results",
                    "ltc$(tag)_zeta$(zeta_tag(zeta)).jld2")
    isfile(path) || (path = joinpath(@__DIR__, "..", "..", "results",
                                     "ltc_zeta$(zeta_tag(zeta)).jld2"))
    p = load(path, "params")
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

# ── Two-pass autonomous rollout (same as mse_comparison) ─────────────────────

function make_seed_iseq(x0::Vector{Float32}, T::Int)::Matrix{Float32}
    M = Matrix{Float32}(undef, T, length(x0))
    for i in 1:T; M[i, :] = x0; end
    return M
end

function rollout_2pass(rollout_fn::Function, cell,
                       x0::Vector{Float32}, t_seq::Vector{Float32})::Matrix{Float32}
    T = length(t_seq); t_span = (t_seq[1], t_seq[end])
    pred  = rollout_fn(cell, zeros(Float32, HIDDEN_DIM),
                       make_seed_iseq(x0, T), t_span, t_seq)
    out   = pred[:, 1:INPUT_DIM]
    I2    = vcat(reshape(x0, 1, INPUT_DIM), out[1:end-1, :])
    pred2 = rollout_fn(cell, zeros(Float32, HIDDEN_DIM), I2, t_span, t_seq)
    return pred2[:, 1:INPUT_DIM]
end

# ── Per-step MSE and recovery analysis ───────────────────────────────────────

function recovery_steps(pred::Matrix{Float32}, target::Matrix{Float32},
                        t_seq::Vector{Float32}; threshold=RECOVERY_THRESHOLD)
    post_idx = findall(t -> t >= T_SWITCH, t_seq)
    isempty(post_idx) && return missing
    for k in post_idx
        step_mse = mean((pred[k,:] .- target[k,:]).^2)
        step_mse < threshold && return k - post_idx[1]   # steps after switch
    end
    return length(post_idx)   # never recovered within window
end

function eval_switch(rollout_fn::Function, cell,
                     t_test::Vector{Float32}, x_test::Matrix{Float32};
                     label::String="")
    pred     = rollout_2pass(rollout_fn, cell, x_test[1,:], t_test)
    pre_idx  = findall(t -> t <  T_SWITCH, t_test)
    post_idx = findall(t -> t >= T_SWITCH, t_test)

    overall_mse = mse_loss_common(pred, x_test)
    pre_mse     = isempty(pre_idx)  ? NaN32 : mse_loss_common(pred[pre_idx,:],  x_test[pre_idx,:])
    post_mse    = isempty(post_idx) ? NaN32 : mse_loss_common(pred[post_idx,:], x_test[post_idx,:])
    rec_steps   = recovery_steps(pred, x_test, t_test)

    return (overall=overall_mse, pre=pre_mse, post=post_mse,
            recovery=rec_steps, pred=pred)
end

# ── Main ──────────────────────────────────────────────────────────────────────

function main()
    results_dir = joinpath(@__DIR__, "..", "..", "results")
    mkpath(results_dir)

    println("Regime-switching experiment")
    println("Models trained on single-ζ; tested on switching trajectories")
    println("Switch point: t=$(T_SWITCH)\n")

    # Use train_zeta = zeta1 (model trained on the pre-switch regime)
    open(joinpath(results_dir, "regime_switch.csv"), "w") do io
        println(io, "zeta1,zeta2,model,overall_mse,pre_mse,post_mse,recovery_steps")

        for (z1, z2) in SWITCH_PAIRS
            println("─"^55)
            @printf("Switch: ζ=%.1f → ζ=%.1f  (model trained on ζ=%.1f)\n", z1, z2, z1)

            t_test, x_test = load_switch_data(z1, z2)

            ltc_cell   = load_ltc_cell(z1)
            node_cell  = load_node_cell(z1)
            ctrnn_cell = load_ctrnn_cell(z1)

            res_ltc   = eval_switch(ltc_rollout,   ltc_cell,   t_test, x_test)
            res_node  = eval_switch(node_rollout,  node_cell,  t_test, x_test)
            res_ctrnn = eval_switch(ctrnn_rollout, ctrnn_cell, t_test, x_test)

            for (name, res) in [("LTC",res_ltc), ("NODE",res_node), ("CTRNN",res_ctrnn)]
                @printf("  %-6s  overall=%.5f  pre=%.5f  post=%.5f  recovery=%s steps\n",
                        name, res.overall, res.pre, res.post,
                        res.recovery === missing ? "N/A" : string(res.recovery))
                println(io, "$(z1),$(z2),$(name),$(res.overall),$(res.pre),$(res.post),$(something(res.recovery, -1))")
            end

            # ── Trajectory plot for this switch pair ──────────────────────────
            p = plot(t_test, x_test[:,1];
                     label="Ground truth", color=:black, linewidth=2,
                     linestyle=:dash,
                     xlabel="t", ylabel="x (position)",
                     title="Regime switch ζ=$(z1)→$(z2)  (autonomous rollout)")
            plot!(p, t_test, res_ltc.pred[:,1];
                  label="LTC",   color=:blue,  linewidth=2)
            plot!(p, t_test, res_node.pred[:,1];
                  label="NODE",  color=:red,   linewidth=2)
            plot!(p, t_test, res_ctrnn.pred[:,1];
                  label="CTRNN", color=:green, linewidth=2)
            vline!(p, [T_SWITCH]; linestyle=:dot, color=:gray,
                   label="switch point", linewidth=1)

            fname = joinpath(results_dir,
                             "regime_switch_$(zeta_tag(z1))to$(zeta_tag(z2)).png")
            savefig(p, fname)
            println("  Plot → $fname")
        end
    end

    # ── Also evaluate LTC with MDL regularisation (adaptive λ) ───────────────
    println("\n" * "="^55)
    println("LTC+MDL (adaptive λ) vs vanilla LTC — post-switch MSE comparison")
    open(joinpath(results_dir, "regime_switch_mdl.csv"), "w") do io
        println(io, "zeta1,zeta2,ltc_post_mse,ltc_mdl_post_mse,improvement")
        for (z1, z2) in SWITCH_PAIRS
            t_test, x_test = load_switch_data(z1, z2)
            ltc_cell     = load_ltc_cell(z1)
            mdl_path     = joinpath(@__DIR__, "..", "..", "results",
                                    "ltc_mdl_adaptive_zeta$(zeta_tag(z1)).jld2")
            if isfile(mdl_path)
                ltc_mdl_cell = load_ltc_cell(z1; mdl_label="mdl_adaptive")
                res_ltc      = eval_switch(ltc_rollout, ltc_cell,     t_test, x_test)
                res_mdl      = eval_switch(ltc_rollout, ltc_mdl_cell, t_test, x_test)
                improvement  = (res_ltc.post - res_mdl.post) / res_ltc.post * 100
                @printf("ζ=%.1f→%.1f  vanilla=%.5f  MDL=%.5f  Δ=%.1f%%\n",
                        z1, z2, res_ltc.post, res_mdl.post, improvement)
                println(io, "$(z1),$(z2),$(res_ltc.post),$(res_mdl.post),$(improvement)")
            else
                println("  Skipping MDL comparison for ζ=$(z1)→$(z2) (checkpoint not found)")
            end
        end
    end

    println("\nSaved results/regime_switch.csv")
    println("Saved results/regime_switch_mdl.csv")
    println("Saved regime_switch_*.png")
end

main()
