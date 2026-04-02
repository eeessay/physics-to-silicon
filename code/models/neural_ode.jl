# models/neural_ode.jl
# Author: Member 2 (YAN)
# EE5311 CA1-21 — Neural ODE baseline
# Struct parameterized over T for ForwardDiff compatibility.

using DifferentialEquations
using SciMLSensitivity


# Struct
struct NeuralODECell{T}
    Wx1::Matrix{T}
    Ui1::Matrix{T}
    b1::Vector{T}
    Wx2::Matrix{T}
    b2::Vector{T}
end

function NeuralODECell(input_dim::Int, hidden_dim::Int)
    scale = 0.1f0
    NeuralODECell{Float32}(
        scale * randn(Float32, hidden_dim, hidden_dim),
        scale * randn(Float32, hidden_dim, input_dim),
        zeros(Float32, hidden_dim),
        scale * randn(Float32, hidden_dim, hidden_dim),
        zeros(Float32, hidden_dim),
    )
end

# Dynamics
function neural_ode_dynamics(x, cell::NeuralODECell, I)
    h = tanh.(cell.Wx1 * x .+ cell.Ui1 * I .+ cell.b1)
    cell.Wx2 * h .+ cell.b2
end

# Rollout using the struct, for inference/validation
function node_rollout(cell::NeuralODECell,
                      x0::Vector{Float32},
                      I_seq::Matrix{Float32},
                      t_span::Tuple{Float32,Float32},
                      t_save::AbstractVector{Float32})::Matrix{Float32}
    p_flat = flatten_node(cell)
    Float32.(node_rollout_p(p_flat, x0, I_seq, t_span, t_save,
                   size(I_seq, 2), length(x0)))
end

# Rollout using flattened parameters (for AD/training)
function node_rollout_p(p_flat::AbstractVector,
                        x0::Vector{Float32},
                        I_seq::Matrix{Float32},
                        t_span::Tuple{Float32,Float32},
                        t_save::AbstractVector{Float32},
                        input_dim::Int,
                        hidden_dim::Int)

    T      = size(I_seq, 1)
    t_grid = collect(range(Float32(t_span[1]), Float32(t_span[2]); length = T))

    function ode!(dx, x, p, t)
        c   = unflatten_node(p, input_dim, hidden_dim)
        idx = clamp(searchsortedfirst(t_grid, Float32(t)), 1, T)
        I   = I_seq[idx, :]
        dx .= neural_ode_dynamics(x, c, I)
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
function flatten_node(cell::NeuralODECell{T}) where T
    vcat(vec(cell.Wx1), vec(cell.Ui1), cell.b1,
         vec(cell.Wx2), cell.b2)
end

function unflatten_node(p::AbstractVector, input_dim::Int, hidden_dim::Int)
    h, i = hidden_dim, input_dim
    idx  = 0
    Wx1 = reshape(p[idx+1 : idx+h*h], h, h); idx += h*h
    Ui1 = reshape(p[idx+1 : idx+h*i], h, i); idx += h*i
    b1  = p[idx+1 : idx+h];                  idx += h
    Wx2 = reshape(p[idx+1 : idx+h*h], h, h); idx += h*h
    b2  = p[idx+1 : idx+h]
    T   = eltype(p)
    NeuralODECell{T}(Wx1, Ui1, b1, Wx2, b2)
end
