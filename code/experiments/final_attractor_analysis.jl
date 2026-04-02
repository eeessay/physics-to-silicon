# attractor_analysis.jl
# Author: YU XINTONG
# EE5311 CA1-21 — Phase-portrait consistency analysis 
#

using DifferentialEquations
using JLD2
using Random
using Statistics
using Printf
using LinearAlgebra

const _HAS_PLOTS = let ok = false
    try
        @eval using Plots
        ok = true
    catch err
        @warn "Plots.jl not available; PNG figures will be skipped." exception=(err, catch_backtrace())
    end
    ok
end

# ── Paths ─────────────────────────────────────────────────────────────────────

const CODE_DIR    = @__DIR__
const ROOT_DIR    = normpath(joinpath(CODE_DIR, ".."))
const MODEL_DIR   = joinpath(CODE_DIR, "code/models")
const RESULTS_DIR = joinpath(ROOT_DIR, "CA1_group21-main/results")
const DATA_DIR    = joinpath(ROOT_DIR, "CA1_group21-main/data")

include(joinpath(MODEL_DIR, "ltc.jl"))
include(joinpath(MODEL_DIR, "ctrnn.jl"))
include(joinpath(MODEL_DIR, "neural_ode.jl"))

# ── Constants ─────────────────────────────────────────────────────────────────

const N_ICS      = 100
const IC_RANGE   = (-2.0f0, 2.0f0)
const ZETA_REF   = 0.3f0
const OMEGA_0    = 1.0f0
const INPUT_DIM  = 2
const HIDDEN_DIM = 8
const SEED       = 42
# Match the project train/test sequence length (100 interleaved points spanning [0,20])
const T_SAVE     = collect(Float32.(range(0.0f0, 20.0f0; length=100)))
const T_SPAN     = (T_SAVE[1], T_SAVE[end])

# ── Utilities ─────────────────────────────────────────────────────────────────

zeta_tag(z::Real) = replace(string(z), "." => "p")

function ensure_results_dir!()
    mkpath(RESULTS_DIR)
end

function checkpoint_path(model::Symbol, zeta::Float32)::String
    tag = zeta_tag(zeta)
    filename = model === :ltc   ? "ltc_zeta$(tag).jld2" :
               model === :node  ? "node_zeta$(tag).jld2" :
               model === :ctrnn ? "ctrnn_zeta$(tag).jld2" :
               error("Unknown model symbol: $model")
    return joinpath(RESULTS_DIR, filename)
end

function require_checkpoint(model::Symbol, zeta::Float32)::String
    path = checkpoint_path(model, zeta)
    isfile(path) || error("Checkpoint not found: $path\nPlease make sure code/train.jl has been run.")
    return path
end

function sample_initial_conditions(n::Int;
                                   range::Tuple{Float32,Float32}=IC_RANGE,
                                   seed::Int=SEED)::Matrix{Float32}
    rng = MersenneTwister(seed)
    lo, hi = range
    ics = rand(rng, Float32, n, 2)
    @. ics = lo + (hi - lo) * ics
    return ics
end

function load_project_dataset(zeta::Float32=ZETA_REF)
    path = joinpath(DATA_DIR, "oscillator_zeta$(zeta_tag(zeta)).jld2")
    isfile(path) || error("Dataset not found: $path\nRun code/data_gen.jl first.")
    return load(path)
end

# ── Hausdorff distance ────────────────────────────────────────────────────────

function hausdorff_distance(A::Matrix{Float32}, B::Matrix{Float32})::Float32
    d_AB = maximum(minimum(sqrt.(sum((A[i:i, :] .- B).^2; dims=2))) for i in axes(A, 1))
    d_BA = maximum(minimum(sqrt.(sum((B[i:i, :] .- A).^2; dims=2))) for i in axes(B, 1))
    return max(d_AB, d_BA)
end

# ── True system ───────────────────────────────────────────────────────────────

function true_oscillator_rhs!(dy, y, p, t)
    ζ, ω0 = p
    x, v = y[1], y[2]
    dy[1] = v
    dy[2] = -2f0 * ζ * ω0 * v - ω0^2 * x
    return nothing
end

function rollout_true(ic::Vector{Float32};
                      zeta::Float32=ZETA_REF,
                      omega0::Float32=OMEGA_0,
                      t_save::Vector{Float32}=T_SAVE)::Matrix{Float32}
    prob = ODEProblem(true_oscillator_rhs!, ic,
                      (Float64(t_save[1]), Float64(t_save[end])),
                      (zeta, omega0))
    sol = solve(prob, Tsit5(); saveat=Float64.(t_save), abstol=1e-8, reltol=1e-8)
    return Float32.(hcat(sol.u...)')
end

# ── Model loading ─────────────────────────────────────────────────────────────

function load_ltc_cell(zeta::Float32=ZETA_REF)::LTCCell
    p = load(require_checkpoint(:ltc, zeta), "params")
    return unflatten(Float32.(p), INPUT_DIM, HIDDEN_DIM)
end

function load_node_cell(zeta::Float32=ZETA_REF)::NeuralODECell
    p = load(require_checkpoint(:node, zeta), "params")
    return unflatten_node(Float32.(p), INPUT_DIM, HIDDEN_DIM)
end

function load_ctrnn_cell(zeta::Float32=ZETA_REF)::CTRNNCell
    p = load(require_checkpoint(:ctrnn, zeta), "params")
    return unflatten_ctrnn(Float32.(p), INPUT_DIM, HIDDEN_DIM)
end

# ── Teacher-forced rollout wrappers ───────────────────────────────────────────
# Use the true trajectory itself as input sequence, consistent with training.
# Hidden state still starts at zero, exactly as in code/train.jl.

function rollout_ltc_teacher_forced(cell::LTCCell,
                                    true_traj::Matrix{Float32};
                                    t_save::Vector{Float32}=T_SAVE)::Matrix{Float32}
    x0 = zeros(Float32, HIDDEN_DIM)
    pred = ltc_rollout(cell, x0, true_traj, (t_save[1], t_save[end]), t_save)
    return pred[:, 1:INPUT_DIM]
end

function rollout_node_teacher_forced(cell::NeuralODECell,
                                     true_traj::Matrix{Float32};
                                     t_save::Vector{Float32}=T_SAVE)::Matrix{Float32}
    x0 = zeros(Float32, HIDDEN_DIM)
    pred = node_rollout(cell, x0, true_traj, (t_save[1], t_save[end]), t_save)
    return pred[:, 1:INPUT_DIM]
end

function rollout_ctrnn_teacher_forced(cell::CTRNNCell,
                                      true_traj::Matrix{Float32};
                                      t_save::Vector{Float32}=T_SAVE)::Matrix{Float32}
    x0 = zeros(Float32, HIDDEN_DIM)
    pred = ctrnn_rollout(cell, x0, true_traj, (t_save[1], t_save[end]), t_save)
    return pred[:, 1:INPUT_DIM]
end

# ── Reference-alignment check (optional LTC sanity check) ────────────────────

function maybe_verify_ltc_alignment(cell::LTCCell)::Nothing
    ref_path = joinpath(RESULTS_DIR, "julia_ltc_reference_output.jld2")
    data_path = joinpath(DATA_DIR, "oscillator_zeta$(zeta_tag(ZETA_REF)).jld2")

    if !isfile(ref_path) || !isfile(data_path)
        println("Alignment check skipped: reference artifact not found.")
        return nothing
    end

    ref_data = load(data_path)
    t_test = Float32.(ref_data["t_test"])
    x_test = Float32.(ref_data["x_test"])
    x0 = zeros(Float32, HIDDEN_DIM)
    pred = ltc_rollout(cell, x0, x_test, (t_test[1], t_test[end]), t_test)
    pred_out = pred[:, 1:INPUT_DIM]
    verify_against_reference(pred_out, ref_path; tol=1.0f-4)
    return nothing
end

# ── Batch simulation ──────────────────────────────────────────────────────────

function simulate_teacher_forced_consistency(ics::Matrix{Float32}; zeta::Float32=ZETA_REF)
    ltc_cell = load_ltc_cell(zeta)
    node_cell = load_node_cell(zeta)
    ctrnn_cell = load_ctrnn_cell(zeta)

    maybe_verify_ltc_alignment(ltc_cell)

    true_trajs  = Vector{Matrix{Float32}}(undef, size(ics, 1))
    ltc_trajs   = Vector{Matrix{Float32}}(undef, size(ics, 1))
    node_trajs  = Vector{Matrix{Float32}}(undef, size(ics, 1))
    ctrnn_trajs = Vector{Matrix{Float32}}(undef, size(ics, 1))

    println("Running teacher-forced phase-portrait consistency for $(size(ics,1)) ICs at ζ=$(zeta)...")
    for k in 1:size(ics, 1)
        ic = vec(ics[k, :])
        true_traj      = rollout_true(ic; zeta=zeta)
        true_trajs[k]  = true_traj
        ltc_trajs[k]   = rollout_ltc_teacher_forced(ltc_cell, true_traj)
        node_trajs[k]  = rollout_node_teacher_forced(node_cell, true_traj)
        ctrnn_trajs[k] = rollout_ctrnn_teacher_forced(ctrnn_cell, true_traj)

        if k % 20 == 0 || k == size(ics,1)
            println("  finished $k / $(size(ics,1)) trajectories")
        end
    end

    return (; true_trajs, ltc_trajs, node_trajs, ctrnn_trajs)
end

stack_points(trajs::Vector{Matrix{Float32}}) = reduce(vcat, trajs)

# ── Metrics ───────────────────────────────────────────────────────────────────

function per_ic_hausdorff(true_trajs::Vector{Matrix{Float32}},
                          model_trajs::Vector{Matrix{Float32}})::Vector{Float32}
    @assert length(true_trajs) == length(model_trajs)
    [hausdorff_distance(true_trajs[k], model_trajs[k]) for k in eachindex(true_trajs)]
end

function global_hausdorff(true_trajs::Vector{Matrix{Float32}},
                          model_trajs::Vector{Matrix{Float32}})::Float32
    return hausdorff_distance(stack_points(true_trajs), stack_points(model_trajs))
end

function summarize_distances(name::String,
                             per_ic::Vector{Float32},
                             global_hd::Float32)
    return (
        model = name,
        mean_hd = mean(per_ic),
        median_hd = median(per_ic),
        std_hd = std(per_ic),
        min_hd = minimum(per_ic),
        max_hd = maximum(per_ic),
        global_hd = global_hd,
    )
end

function metric_mse(A::Matrix{Float32}, B::Matrix{Float32})::Float32
    mean((A .- B).^2)
end

function per_ic_mse(true_trajs::Vector{Matrix{Float32}},
                    model_trajs::Vector{Matrix{Float32}})::Vector{Float32}
    [metric_mse(true_trajs[k], model_trajs[k]) for k in eachindex(true_trajs)]
end

# ── Writers ───────────────────────────────────────────────────────────────────

function save_summary_csv(path::String, summaries, mean_mses::Dict{String,Float32})
    open(path, "w") do io
        println(io, "model,mean_hd,median_hd,std_hd,min_hd,max_hd,global_hd,mean_mse")
        for s in summaries
            mmse = mean_mses[s.model]
            @printf(io, "%s,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                    s.model, s.mean_hd, s.median_hd, s.std_hd,
                    s.min_hd, s.max_hd, s.global_hd, mmse)
        end
    end
end

function save_per_ic_csv(path::String,
                         ltc_hd::Vector{Float32}, node_hd::Vector{Float32}, ctrnn_hd::Vector{Float32},
                         ltc_mse::Vector{Float32}, node_mse::Vector{Float32}, ctrnn_mse::Vector{Float32})
    open(path, "w") do io
        println(io, "ic_id,ltc_hd,node_hd,ctrnn_hd,ltc_mse,node_mse,ctrnn_mse")
        for k in eachindex(ltc_hd)
            @printf(io, "%d,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                    k, ltc_hd[k], node_hd[k], ctrnn_hd[k], ltc_mse[k], node_mse[k], ctrnn_mse[k])
        end
    end
end

function save_ics_csv(path::String, ics::Matrix{Float32})
    open(path, "w") do io
        println(io, "ic_id,x0,v0")
        for k in 1:size(ics,1)
            @printf(io, "%d,%.8f,%.8f\n", k, ics[k,1], ics[k,2])
        end
    end
end

function ranking_string(summaries, field::Symbol)
    ordered = sort(collect(summaries), by = s -> getfield(s, field))
    join(["$(s.model) ($(round(getfield(s, field); digits=4)))" for s in ordered], " < ")
end

function save_report(path::String, summaries, mean_mses::Dict{String,Float32};
                     zeta::Float32=ZETA_REF, n_ics::Int=N_ICS)
    open(path, "w") do io
        println(io, "EE5311 CA1-21 — Phase-portrait consistency report")
        println(io, "="^64)
        println(io, "Reference damping ratio ζ = $zeta")
        println(io, "Number of random initial conditions = $n_ics")
        println(io, "Initial-condition box = [$(IC_RANGE[1]), $(IC_RANGE[2])]^2")
        println(io)
        println(io, "Evaluation protocol")
        println(io, "-------------------")
        println(io, "For each sampled initial condition, the true damped oscillator trajectory")
        println(io, "is generated first. That true trajectory is then fed into each trained")
        println(io, "model as the input sequence (teacher forcing), consistent with code/train.jl.")
        println(io, "This evaluates phase-portrait / trajectory-geometry consistency rather")
        println(io, "than strict autonomous attractor reconstruction.")
        println(io)
        println(io, "Summary table")
        println(io, "-------------")
        @printf(io, "%-12s %-12s %-12s %-12s %-12s %-12s\n",
                "Model", "Mean HD", "Median HD", "Std HD", "Global HD", "Mean MSE")
        for s in summaries
            @printf(io, "%-12s %-12.6f %-12.6f %-12.6f %-12.6f %-12.6f\n",
                    s.model, s.mean_hd, s.median_hd, s.std_hd, s.global_hd, mean_mses[s.model])
        end
        println(io)
        println(io, "Rankings (lower is better)")
        println(io, "--------------------------")
        println(io, "Per-trajectory mean Hausdorff ranking : " * ranking_string(summaries, :mean_hd))
        println(io, "Per-trajectory median Hausdorff ranking: " * ranking_string(summaries, :median_hd))
        println(io, "Global point-cloud Hausdorff ranking  : " * ranking_string(summaries, :global_hd))
        println(io)
        println(io, "Interpretation")
        println(io, "--------------")
        println(io, "- mean/median HD reflects trajectory-level geometric consistency.")
        println(io, "- global HD is reported for completeness but can be optimistic when")
        println(io, "  many trajectories overlap in a similar region.")
        println(io, "- Use the saved phase-portrait figures together with HD/MSE to discuss")
        println(io, "  which model preserves the true damped-oscillator geometry most faithfully.")
    end
end

# ── Plot helpers ──────────────────────────────────────────────────────────────

function axis_limits(true_trajs::Vector{Matrix{Float32}})
    all_true = stack_points(true_trajs)
    xmin = minimum(all_true[:,1]); xmax = maximum(all_true[:,1])
    ymin = minimum(all_true[:,2]); ymax = maximum(all_true[:,2])
    padx = 0.05f0 * max(1f-3, xmax - xmin)
    pady = 0.05f0 * max(1f-3, ymax - ymin)
    return (xmin - padx, xmax + padx), (ymin - pady, ymax + pady)
end

function plot_phase_portraits(true_trajs, node_trajs, ctrnn_trajs, ltc_trajs; path::String)
    _HAS_PLOTS || return nothing

    xlims_, ylims_ = axis_limits(true_trajs)
    plt = plot(layout=(2,2), size=(1100,900), legend=false)
    groups = [
        (true_trajs,  "True system"),
        (node_trajs,  "Neural ODE"),
        (ctrnn_trajs, "CTRNN"),
        (ltc_trajs,   "LTC"),
    ]

    for (idx, (trajs, ttl)) in enumerate(groups)
        for traj in trajs
            plot!(plt[idx], traj[:,1], traj[:,2], lw=1.0, alpha=0.40, color=:steelblue)
        end
        scatter!(plt[idx], [0.0], [0.0], markersize=3, color=:black)
        xlabel!(plt[idx], "x")
        ylabel!(plt[idx], "v")
        title!(plt[idx], ttl)
        xlims!(plt[idx], xlims_)
        ylims!(plt[idx], ylims_)
        plot!(plt[idx], aspect_ratio = :equal)
    end

    savefig(plt, path)
    return nothing
end

function pick_demo_indices(n::Int; k::Int=10)
    k = min(k, n)
    return round.(Int, range(1, n, length=k))
end

function plot_overlay_vs_true(true_trajs, model_trajs, title_str::String, path::String)
    _HAS_PLOTS || return nothing

    xlims_, ylims_ = axis_limits(true_trajs)
    idxs = pick_demo_indices(length(true_trajs); k=12)
    plt = plot(size=(760,640), title=title_str, xlabel="x", ylabel="v",
               legend=:outertopright, aspect_ratio=:equal,
               xlims=xlims_, ylims=ylims_)

    first_true = true
    first_model = true
    for idx in idxs
        tt = true_trajs[idx]
        mt = model_trajs[idx]
        plot!(plt, tt[:,1], tt[:,2], lw=2.0, alpha=0.75, color=:steelblue,
              label = first_true ? "True trajectory" : "")
        plot!(plt, mt[:,1], mt[:,2], lw=1.8, alpha=0.75, color=:crimson, linestyle=:dash,
              label = first_model ? "Model output" : "")
        scatter!(plt, [tt[1,1]], [tt[1,2]], markersize=3, color=:black,
                 label = idx == idxs[1] ? "Initial state" : "")
        first_true = false
        first_model = false
    end

    savefig(plt, path)
    return nothing
end

function plot_per_ic_scatter(ltc_hd, node_hd, ctrnn_hd; path::String)
    _HAS_PLOTS || return nothing
    idx = collect(1:length(ltc_hd))
    plt = plot(size=(900,600), xlabel="IC index", ylabel="Hausdorff distance",
               title="Per-initial-condition Hausdorff distance", legend=:topright)
    scatter!(plt, idx, ltc_hd, markersize=3, label="LTC")
    scatter!(plt, idx, node_hd, markersize=3, label="Neural ODE")
    scatter!(plt, idx, ctrnn_hd, markersize=3, label="CTRNN")
    savefig(plt, path)
    return nothing
end

function plot_best_model_map(ics::Matrix{Float32}, ltc_hd, node_hd, ctrnn_hd; path::String)
    _HAS_PLOTS || return nothing

    best = Vector{String}(undef, size(ics,1))
    for i in 1:size(ics,1)
        vals = Dict("LTC" => ltc_hd[i], "Neural ODE" => node_hd[i], "CTRNN" => ctrnn_hd[i])
        best[i] = first(sort(collect(keys(vals)), by = k -> vals[k]))
    end

    plt = plot(size=(700,650), xlabel="Initial position x₀", ylabel="Initial velocity v₀",
               title="Best-performing model by initial condition", legend=:outertopright)
    for label in ["LTC", "Neural ODE", "CTRNN"]
        idxs = findall(==(label), best)
        if !isempty(idxs)
            scatter!(plt, ics[idxs,1], ics[idxs,2], markersize=4, alpha=0.80, label=label)
        end
    end
    savefig(plt, path)
    return nothing
end

# ── Main ──────────────────────────────────────────────────────────────────────

function main()
    ensure_results_dir!()

    println("="^72)
    println("EE5311 CA1-21 — Phase-portrait consistency analysis ")
    println("Root directory   : $ROOT_DIR")
    println("Results directory: $RESULTS_DIR")
    println("ZETA_REF         : $ZETA_REF")
    println("N_ICS            : $N_ICS")
    println("="^72)

    # Basic project sanity checks
    _ = load_project_dataset(ZETA_REF)

    ics = sample_initial_conditions(N_ICS)
    save_ics_csv(joinpath(RESULTS_DIR, "attractor_initial_conditions.csv"), ics)

    sims = simulate_teacher_forced_consistency(ics; zeta=ZETA_REF)
    true_trajs  = sims.true_trajs
    ltc_trajs   = sims.ltc_trajs
    node_trajs  = sims.node_trajs
    ctrnn_trajs = sims.ctrnn_trajs

    ltc_hd   = per_ic_hausdorff(true_trajs, ltc_trajs)
    node_hd  = per_ic_hausdorff(true_trajs, node_trajs)
    ctrnn_hd = per_ic_hausdorff(true_trajs, ctrnn_trajs)

    ltc_mse   = per_ic_mse(true_trajs, ltc_trajs)
    node_mse  = per_ic_mse(true_trajs, node_trajs)
    ctrnn_mse = per_ic_mse(true_trajs, ctrnn_trajs)

    ltc_global   = global_hausdorff(true_trajs, ltc_trajs)
    node_global  = global_hausdorff(true_trajs, node_trajs)
    ctrnn_global = global_hausdorff(true_trajs, ctrnn_trajs)

    summaries = [
        summarize_distances("LTC", ltc_hd, ltc_global),
        summarize_distances("NeuralODE", node_hd, node_global),
        summarize_distances("CTRNN", ctrnn_hd, ctrnn_global),
    ]
    mean_mses = Dict(
        "LTC" => mean(ltc_mse),
        "NeuralODE" => mean(node_mse),
        "CTRNN" => mean(ctrnn_mse),
    )

    summary_csv = joinpath(RESULTS_DIR, "attractor_hausdorff_summary.csv")
    per_ic_csv  = joinpath(RESULTS_DIR, "attractor_per_ic_distances.csv")
    report_txt  = joinpath(RESULTS_DIR, "attractor_analysis_report.txt")

    save_summary_csv(summary_csv, summaries, mean_mses)
    save_per_ic_csv(per_ic_csv, ltc_hd, node_hd, ctrnn_hd, ltc_mse, node_mse, ctrnn_mse)
    save_report(report_txt, summaries, mean_mses)

    plot_phase_portraits(true_trajs, node_trajs, ctrnn_trajs, ltc_trajs;
                         path=joinpath(RESULTS_DIR, "attractor_phase_portraits.png"))
    plot_overlay_vs_true(true_trajs, ltc_trajs,
                         "True system vs LTC (teacher-forced overlay)",
                         joinpath(RESULTS_DIR, "attractor_overlay_ltc_vs_true.png"))
    plot_overlay_vs_true(true_trajs, node_trajs,
                         "True system vs Neural ODE (teacher-forced overlay)",
                         joinpath(RESULTS_DIR, "attractor_overlay_node_vs_true.png"))
    plot_overlay_vs_true(true_trajs, ctrnn_trajs,
                         "True system vs CTRNN (teacher-forced overlay)",
                         joinpath(RESULTS_DIR, "attractor_overlay_ctrnn_vs_true.png"))
    plot_per_ic_scatter(ltc_hd, node_hd, ctrnn_hd;
                        path=joinpath(RESULTS_DIR, "attractor_hausdorff_scatter.png"))
    plot_best_model_map(ics, ltc_hd, node_hd, ctrnn_hd;
                        path=joinpath(RESULTS_DIR, "attractor_best_model_by_ic.png"))

    println("\nSummary table (lower is better):")
    @printf("%-12s %-12s %-12s %-12s %-12s %-12s\n",
            "Model", "Mean HD", "Median HD", "Std HD", "Global HD", "Mean MSE")
    for s in summaries
        @printf("%-12s %-12.6f %-12.6f %-12.6f %-12.6f %-12.6f\n",
                s.model, s.mean_hd, s.median_hd, s.std_hd, s.global_hd, mean_mses[s.model])
    end

    println("\nSaved outputs:")
    println("  - " * summary_csv)
    println("  - " * per_ic_csv)
    println("  - " * report_txt)
    println("  - " * joinpath(RESULTS_DIR, "attractor_initial_conditions.csv"))
    if _HAS_PLOTS
        println("  - " * joinpath(RESULTS_DIR, "attractor_phase_portraits.png"))
        println("  - " * joinpath(RESULTS_DIR, "attractor_overlay_ltc_vs_true.png"))
        println("  - " * joinpath(RESULTS_DIR, "attractor_overlay_node_vs_true.png"))
        println("  - " * joinpath(RESULTS_DIR, "attractor_overlay_ctrnn_vs_true.png"))
        println("  - " * joinpath(RESULTS_DIR, "attractor_hausdorff_scatter.png"))
        println("  - " * joinpath(RESULTS_DIR, "attractor_best_model_by_ic.png"))
    end

    println("\nPhase-portrait consistency analysis complete.")
    return nothing
end

main()
