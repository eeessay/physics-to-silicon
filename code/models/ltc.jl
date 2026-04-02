# models/ltc.jl
# Author: Member 1 (Chang)
# EE5311 CA1-21 — Liquid Time-Constant Network
#
# v3 (AD fix): struct fields parameterized over element type T so ForwardDiff
#   Dual numbers can flow through unflatten → LTCCell without hitting the
#   Float32(::Dual) conversion error.

using DifferentialEquations
using SciMLSensitivity
using JLD2
using LinearAlgebra

# ── Activation ────────────────────────────────────────────────────────────────

softplus(x::Real) = log(one(x) + exp(x))

# ── Struct (parameterized) ────────────────────────────────────────────────────

struct LTCCell{T}
    Wτ::Matrix{T}
    Uτ::Matrix{T}
    bτ::Vector{T}
    τ₀::Vector{T}
    Wf::Matrix{T}
    Uf::Matrix{T}
    bf::Vector{T}
end

function LTCCell(input_dim::Int, hidden_dim::Int; tau0::Float32 = 1.0f0)
    scale = 0.1f0
    LTCCell{Float32}(
        scale * randn(Float32, hidden_dim, hidden_dim),
        scale * randn(Float32, hidden_dim, input_dim),
        zeros(Float32, hidden_dim),
        fill(tau0, hidden_dim),
        scale * randn(Float32, hidden_dim, hidden_dim),
        scale * randn(Float32, hidden_dim, input_dim),
        zeros(Float32, hidden_dim),
    )
end

# ── Forward pass ──────────────────────────────────────────────────────────────

function tau(cell::LTCCell, x, I)
    cell.τ₀ .+ softplus.(cell.Wτ * x .+ cell.Uτ * I .+ cell.bτ)
end

function f_drift(cell::LTCCell, x, I)
    tanh.(cell.Wf * x .+ cell.Uf * I .+ cell.bf)
end

function ltc_dynamics(x, cell::LTCCell, I)
    τ_val = tau(cell, x, I)
    f_val = f_drift(cell, x, I)
    @. -x / τ_val + f_val
end

# ── Rollout (struct interface, for inference/validation) ──────────────────────

function ltc_rollout(cell::LTCCell,
                     x0::Vector{Float32},
                     I_seq::Matrix{Float32},
                     t_span::Tuple{Float32,Float32},
                     t_save::AbstractVector{Float32})::Matrix{Float32}
    p_flat = flatten(cell)
    Float32.(ltc_rollout_p(p_flat, x0, I_seq, t_span, t_save,
                  size(I_seq, 2), length(x0)))
end

# ── Rollout (flat-param interface, Zygote-differentiable) ─────────────────────

function ltc_rollout_p(p_flat::AbstractVector,
                       x0::Vector{Float32},
                       I_seq::Matrix{Float32},
                       t_span::Tuple{Float32,Float32},
                       t_save::AbstractVector{Float32},
                       input_dim::Int,
                       hidden_dim::Int)

    T      = size(I_seq, 1)
    t_grid = collect(range(Float32(t_span[1]), Float32(t_span[2]); length = T))

    function ode!(dx, x, p, t)
        c   = unflatten(p, input_dim, hidden_dim)   # returns LTCCell{eltype(p)}
        idx = clamp(searchsortedfirst(t_grid, Float32(t)), 1, T)
        I   = I_seq[idx, :]
        dx .= ltc_dynamics(x, c, I)
    end

    prob = ODEProblem(ode!, x0,
                      (Float64(t_span[1]), Float64(t_span[2])),
                      p_flat)
    sol  = solve(prob, Tsit5();
                 saveat   = Float64.(t_save),
                 abstol   = 1e-6, reltol  = 1e-6,
                 sensealg = ForwardDiffSensitivity())
    hcat(sol.u...)'
end

# ── Parameter serialization ───────────────────────────────────────────────────

function flatten(cell::LTCCell{T}) where T
    vcat(vec(cell.Wτ), vec(cell.Uτ), cell.bτ, cell.τ₀,
         vec(cell.Wf), vec(cell.Uf), cell.bf)
end

# unflatten returns LTCCell{eltype(p)} — works for Float32 and Dual alike
function unflatten(p::AbstractVector, input_dim::Int, hidden_dim::Int)
    h, i = hidden_dim, input_dim
    idx  = 0
    Wτ = reshape(p[idx+1 : idx+h*h], h, h); idx += h*h
    Uτ = reshape(p[idx+1 : idx+h*i], h, i); idx += h*i
    bτ = p[idx+1 : idx+h];                  idx += h
    τ₀ = p[idx+1 : idx+h];                  idx += h
    Wf = reshape(p[idx+1 : idx+h*h], h, h); idx += h*h
    Uf = reshape(p[idx+1 : idx+h*i], h, i); idx += h*i
    bf = p[idx+1 : idx+h]
    T  = eltype(p)
    LTCCell{T}(Wτ, Uτ, bτ, τ₀, Wf, Uf, bf)
end

# ── Loss ──────────────────────────────────────────────────────────────────────

mse_loss(pred::AbstractMatrix, target::AbstractMatrix) = mean((pred .- target).^2)

# ── Alignment verification ────────────────────────────────────────────────────

function verify_against_reference(pred::Matrix{Float32},
                                   ref_path::String = "results/julia_ltc_reference_output.jld2";
                                   tol::Float32 = 1.0f-4)
    ref     = load(ref_path, "pred")
    max_err = maximum(abs.(pred .- ref))
    passed  = max_err < tol
    println("Alignment check: max_err = $(round(max_err; digits=6))  →  $(passed ? "PASS ✓" : "FAIL ✗")")
    return passed
end
