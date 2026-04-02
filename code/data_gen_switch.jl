# data_gen_switch.jl
# Author: Member 1 (Chang)
# EE5311 CA1-21 — Regime-switching dataset generation
#
# Generates trajectories where ζ switches at t=T_SWITCH.
# Uses two-segment integration to avoid discontinuity issues with adaptive solvers.
#
# Output: data/switch_zeta{A}to{B}.jld2

using DifferentialEquations
using JLD2
using Random

const OMEGA_0  = 1.0f0
const T_END    = 20.0f0
const T_SWITCH = 10.0f0
const N_POINTS = 200
const SEED     = 42

const SWITCH_PAIRS = [
    (0.1f0, 1.2f0),
    (1.2f0, 0.1f0),
    (0.3f0, 0.8f0),
    (0.8f0, 0.3f0),
]

Random.seed!(SEED)
zeta_tag(z::Float32) = replace(string(z), "." => "p")

function damped_osc!(du, u, p, t)
    ζ, ω = p
    du[1] = u[2]
    du[2] = -2ζ * ω * u[2] - ω^2 * u[1]
end

# Two-segment integration: avoids discontinuity at T_SWITCH
function generate_switch(z1::Float32, z2::Float32)
    u0 = Float32[1.0f0, 0.0f0]

    # Segment 1: t ∈ [0, T_SWITCH], ζ = z1
    n1 = round(Int, N_POINTS * T_SWITCH / T_END)  # 100
    t1 = collect(range(0.0f0, T_SWITCH; length=n1))
    sol1 = solve(
        ODEProblem(damped_osc!, u0, (0.0, Float64(T_SWITCH)), (z1, OMEGA_0)),
        Tsit5(); saveat=Float64.(t1), abstol=1e-8, reltol=1e-8)
    traj1 = Float32.(hcat(sol1.u...)')  # (n1, 2)

    # Segment 2: t ∈ (T_SWITCH, T_END], ζ = z2, IC = end of segment 1
    n2 = N_POINTS - n1  # 100
    t2 = collect(range(T_SWITCH, T_END; length=n2+1))[2:end]  # exclude T_SWITCH
    sol2 = solve(
        ODEProblem(damped_osc!, sol1.u[end], (Float64(T_SWITCH), Float64(T_END)), (z2, OMEGA_0)),
        Tsit5(); saveat=Float64.(t2), abstol=1e-8, reltol=1e-8)
    traj2 = Float32.(hcat(sol2.u...)')  # (n2, 2)

    t_full    = vcat(Float32.(t1), Float32.(t2))   # (200,)
    traj_full = vcat(traj1, traj2)                  # (200, 2)

    # Interleaved split (consistent with main dataset)
    train_idx = 1:2:N_POINTS
    test_idx  = 2:2:N_POINTS

    return (Float32.(t_full[train_idx]), traj_full[train_idx, :],
            Float32.(t_full[test_idx]),  traj_full[test_idx,  :],
            t_full, traj_full)
end

function main()
    mkpath("data")
    for (z1, z2) in SWITCH_PAIRS
        println("Generating switch ζ=$(z1) → ζ=$(z2)...")
        t_train, x_train, t_test, x_test, t_full, x_full = generate_switch(z1, z2)

        fname = "data/switch_zeta$(zeta_tag(z1))to$(zeta_tag(z2)).jld2"
        jldsave(fname;
            t_train, x_train, t_test, x_test,
            t_full, x_full,
            zeta1=z1, zeta2=z2, t_switch=T_SWITCH)
        println("  Saved → $fname  ($(length(t_train)) train, $(length(t_test)) test pts)")
    end
    println("\nDone.")
end

main()
