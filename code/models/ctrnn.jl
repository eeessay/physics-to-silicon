# models/ctrnn.jl
# Author: Member 2 (YAN)
# EE5311 CA1-21 — CTRNN with learnable fixed time constants
# Struct parameterized over T for ForwardDiff compatibility.

using DifferentialEquations
using SciMLSensitivity

# Struct
struct CTRNNCell{T}
    W::Matrix{T}
    U::Matrix{T}
    b::Vector{T}
    log_τ::Vector{T}
end

function CTRNNCell(input_dim::Int, hidden_dim::Int; tau_init::Float32 = 1.0f0)
    scale = 0.1f0
    CTRNNCell{Float32}(
        scale * randn(Float32, hidden_dim, hidden_dim),
        scale * randn(Float32, hidden_dim, input_dim),
        zeros(Float32, hidden_dim),
        fill(log(tau_init), hidden_dim),
    )
end

# Dynamics
function ctrnn_dynamics(x, cell::CTRNNCell, I)
    τ     = exp.(cell.log_τ)
    drift = tanh.(cell.W * x .+ cell.U * I .+ cell.b)
    @. -x / τ + drift
end

# Rollout using the struct, for inference/validation
function ctrnn_rollout(cell::CTRNNCell,
                       x0::Vector{Float32},
                       I_seq::Matrix{Float32},
                       t_span::Tuple{Float32,Float32},
                       t_save::AbstractVector{Float32})::Matrix{Float32}
    p_flat = flatten_ctrnn(cell)
    Float32.(ctrnn_rollout_p(p_flat, x0, I_seq, t_span, t_save,
                    size(I_seq, 2), length(x0)))
end

# Rollout using flattened parameters (for AD/training)
function ctrnn_rollout_p(p_flat::AbstractVector,
                         x0::Vector{Float32},
                         I_seq::Matrix{Float32},
                         t_span::Tuple{Float32,Float32},
                         t_save::AbstractVector{Float32},
                         input_dim::Int,
                         hidden_dim::Int)

    T      = size(I_seq, 1)
    t_grid = collect(range(Float32(t_span[1]), Float32(t_span[2]); length = T))

    function ode!(dx, x, p, t)
        c   = unflatten_ctrnn(p, input_dim, hidden_dim)
        idx = clamp(searchsortedfirst(t_grid, Float32(t)), 1, T)
        I   = I_seq[idx, :]
        dx .= ctrnn_dynamics(x, c, I)
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

# Flatten/unflatten helpers
function flatten_ctrnn(cell::CTRNNCell{T}) where T
    vcat(vec(cell.W), vec(cell.U), cell.b, cell.log_τ)
end

function unflatten_ctrnn(p::AbstractVector, input_dim::Int, hidden_dim::Int)
    h, i = hidden_dim, input_dim
    idx  = 0
    W     = reshape(p[idx+1 : idx+h*h], h, h); idx += h*h
    U     = reshape(p[idx+1 : idx+h*i], h, i); idx += h*i
    b     = p[idx+1 : idx+h];                  idx += h
    log_τ = p[idx+1 : idx+h]
    T_el  = eltype(p)
    CTRNNCell{T_el}(W, U, b, log_τ)
end
