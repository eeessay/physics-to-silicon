# profiling/julia_profiling.jl
# Author: Member 1 (Chang)
# EE5311 CA1-21 — Julia profiling: inference time, memory, type stability

using InteractiveUtils   # required for @code_warntype in script mode
using BenchmarkTools
using JLD2
using Printf
using Statistics

include(joinpath(@__DIR__, "../models/ltc.jl"))

# ── Setup ─────────────────────────────────────────────────────────────────────

const INPUT_DIM  = 2
const HIDDEN_DIM = 8    # match train.jl

const T_STEPS    = 160
const x0         = zeros(Float32, HIDDEN_DIM)
const I_seq      = randn(Float32, T_STEPS, INPUT_DIM)
const t_span     = (0.0f0, 16.0f0)
const t_save     = Float32.(range(0.0, 16.0, length = T_STEPS))

# Load trained model if available, otherwise use random weights
function load_ltc()
    ckpt = "results/ltc_zeta0p3.jld2"
    if isfile(ckpt)
        p    = load(ckpt, "params")
        cell = unflatten(p, INPUT_DIM, HIDDEN_DIM)
        println("Loaded trained LTC from $ckpt")
        return cell
    else
        println("No checkpoint found, using random LTC weights")
        return LTCCell(INPUT_DIM, HIDDEN_DIM)
    end
end

ltc_model = load_ltc()

# ── 1. Type stability check ───────────────────────────────────────────────────

println("\n", "=" ^ 60)
println("1. TYPE STABILITY CHECK")
println("=" ^ 60)

println("\n--- tau(cell, x, I) ---")
x_test = zeros(Float32, HIDDEN_DIM)
I_test = zeros(Float32, INPUT_DIM)
@code_warntype tau(ltc_model, x_test, I_test)

println("\n--- ltc_dynamics(x, cell, I) ---")
@code_warntype ltc_dynamics(x_test, ltc_model, I_test)

# ── 2. Inference time benchmark ───────────────────────────────────────────────

println("\n", "=" ^ 60)
println("2. INFERENCE TIME (BenchmarkTools, min of 100 samples)")
println("=" ^ 60)

println("\nWarming up...")
ltc_rollout(ltc_model, x0, I_seq, t_span, t_save)  # trigger compilation

println("Benchmarking LTC rollout...")
b_ltc = @benchmark ltc_rollout($ltc_model, $x0, $I_seq, $t_span, $t_save) samples=100 evals=1

println("\nLTC rollout benchmark:")
show(stdout, MIME"text/plain"(), b_ltc)

# ── 3. Memory allocation tracking ────────────────────────────────────────────

println("\n\n", "=" ^ 60)
println("3. MEMORY ALLOCATION (@time)")
println("=" ^ 60)

println("\nLTC rollout:")
@time ltc_rollout(ltc_model, x0, I_seq, t_span, t_save)

# ── 4. Summary table ──────────────────────────────────────────────────────────

println("\n", "=" ^ 60)
println("4. SUMMARY")
println("=" ^ 60)
@printf("%-15s  %12s  %10s  %10s\n", "Model", "Median (ms)", "Min (ms)", "Allocs")
@printf("%-15s  %12.3f  %10.3f  %10d\n",
        "LTC",
        median(b_ltc).time / 1e6,
        minimum(b_ltc).time / 1e6,
        b_ltc.allocs)
println("\nNote: Neural ODE and CTRNN rows will be added by Member 2.")

# ── 5. Save results ───────────────────────────────────────────────────────────

mkpath("results")
open("results/julia_profiling_results.csv", "w") do f
    println(f, "model,median_ms,min_ms,allocs")
    @printf(f, "LTC,%.3f,%.3f,%d\n",
            median(b_ltc).time / 1e6,
            minimum(b_ltc).time / 1e6,
            b_ltc.allocs)
end
println("\nResults saved → results/julia_profiling_results.csv")
