# data_gen.jl
# Author: Member 1 (Chang)
# EE5311 CA1-21 — Damped oscillator dataset generation
# Run once; commit output to data/ for all members to use
#
# v2: interleaved train/test split.
#   Even indices (0,2,4,…) → train  (N/2 = 100 pts, t ∈ [0,20] interleaved)
#   Odd  indices (1,3,5,…) → test   (N/2 = 100 pts, same range)
#   This ensures test points span the full trajectory, making evaluation
#   meaningful for all ζ values, including fast-decaying ones (ζ≥0.5).

using DifferentialEquations
using JLD2
using Random

# ── Parameters ────────────────────────────────────────────────────────────────

const ZETAS      = [0.1f0, 0.3f0, 0.5f0, 0.8f0, 1.2f0]
const OMEGA_0    = 1.0f0
const T_SPAN     = (0.0f0, 20.0f0)
const N_POINTS   = 200
const NOISE_STD  = 0.01f0
const SEED       = 42

Random.seed!(SEED)

# ── ODE definition ─────────────────────────────────────────────────────────────

function damped_oscillator!(du::AbstractVector{T}, u::AbstractVector{T},
                            p::NamedTuple, t::Real) where T
    ζ, ω₀ = p.zeta, p.omega0
    du[1] = u[2]
    du[2] = -2ζ * ω₀ * u[2] - ω₀^2 * u[1]
end

# ── Generate one trajectory ────────────────────────────────────────────────────

function generate_trajectory(zeta::Float32;
                              u0::Vector{Float32} = [1.0f0, 0.0f0],
                              add_noise::Bool = false)
    p    = (zeta = zeta, omega0 = OMEGA_0)
    t    = range(T_SPAN[1], T_SPAN[2], length = N_POINTS)
    prob = ODEProblem(damped_oscillator!, u0, T_SPAN, p)
    sol  = solve(prob, Tsit5(); saveat = collect(t), abstol = 1e-8, reltol = 1e-8)

    traj = Float32.(hcat(sol.u...)')   # (N_POINTS, 2)

    if add_noise
        traj .+= NOISE_STD * randn(Float32, size(traj))
    end

    return collect(t), traj
end

# ── Interleaved train / test split ────────────────────────────────────────────
# even indices (1,3,5,… in 1-based Julia) → train
# odd  indices (2,4,6,…)                  → test
# Both sets span the full [0,20] range.

function split_trajectory(t::AbstractVector, traj::Matrix{Float32})
    train_idx = 1:2:length(t)   # 1,3,5,… → 100 points
    test_idx  = 2:2:length(t)   # 2,4,6,… → 100 points

    t_train = Float32.(t[train_idx])
    t_test  = Float32.(t[test_idx])
    x_train = traj[train_idx, :]
    x_test  = traj[test_idx,  :]

    return t_train, x_train, t_test, x_test
end

# ── Main ──────────────────────────────────────────────────────────────────────

function main()
    mkpath("data")

    for zeta in ZETAS
        println("Generating ζ = $zeta …")

        t, traj       = generate_trajectory(zeta)
        t_train, x_train, t_test, x_test = split_trajectory(t, traj)

        _, traj_noisy = generate_trajectory(zeta; add_noise = true)
        _, x_train_noisy, _, x_test_noisy = split_trajectory(t, traj_noisy)

        fname = "data/oscillator_zeta$(replace(string(zeta), "." => "p")).jld2"
        jldsave(fname;
            t_train, x_train, t_test, x_test,
            x_train_noisy, x_test_noisy,
            zeta, omega0 = OMEGA_0)

        println("  Saved → $fname  (train: $(length(t_train)) pts interleaved, " *
                "test: $(length(t_test)) pts interleaved)")
    end

    println("\nDone. Files in data/:")
    for f in readdir("data"; join = true)
        endswith(f, ".jld2") && println("  $f")
    end
end

main()
