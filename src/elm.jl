struct ELM{T <: Number}
    n_features::Int
    n_outputs::Int
    hidden_layer::MutableHiddenLayer{T}

    function ELM{T}(n_features::Int,
                    n_outputs::Int) where {T <: Number}
        hidden_layer = MutableHiddenLayer(T)
        new{T}(n_features, n_outputs, hidden_layer)
    end

    function ELM(n_features::Int,
                 n_outputs::Int)
        ELM{Float64}(n_features, n_outputs)
    end
end

function add_neurons!(elm::T₁,
                      n_neurons::Int,
                      activation_function::T₂) where {T₁ <: ELM,
                                                      T₂ <: ActivationFunction}
    param(::HiddenLayer{T}) where {T} = T

    @assert begin
        isnothing(elm.hidden_layer.HH) &&
        isnothing(elm.hidden_layer.TH)
    end "Call clear! on ELM before adding neurons."

    neurons = Neurons(param(elm.hidden_layer), n_neurons,
                      elm.n_features, activation_function)
    elm.hidden_layer.n_neurons += n_neurons
    push!(elm.hidden_layer.neurons, neurons)
    elm
end

function add_data!(elm::T₁,
                   samples::T₂,
                   targets::T₃,
                   sample_weights::T₄;
                   batch_size::Int = 1000) where {T₁ <: ELM,
                                                  T₂ <: AbstractMatrix,
                                                  T₃ <: AbstractMatrix,
                                                  T₄ <: AbstractVector}
    @assert elm.hidden_layer.n_neurons > 0 "Add neurons to ELM before adding data."

    if !is_initialized(elm.hidden_layer)
        initialize!(elm.hidden_layer, elm.n_outputs)
    end

    X = samples
    T = targets
    ψ = sample_weights
    Dₓ, Nₓ = size(X)
    Qₜ, Nₜ = size(T)

    @assert Dₓ == elm.n_features "Input dimensionality mismatch."
    @assert Qₜ == elm.n_outputs "Output dimensionality mismatch."
    @assert Nₓ == Nₜ == length(ψ) "Sample count mismatch."

    N = Nₓ

    for batch in partition_range(1:N, batch_size)
        X₀ = X[:,batch]
        T₀ = T[:,batch]
        ψ₀ = ψ[batch]
        add_batch!(elm, X₀, T₀, ψ₀)
    end

    elm
end

function add_data!(elm::T₁,
                   samples::T₂,
                   targets::T₃;
                   batch_size::Int = 1000) where {T₁ <: ELM,
                                                  T₂ <: AbstractMatrix,
                                                  T₃ <: AbstractMatrix}
    N = last(size(samples))
    sample_weights = [1.0 for n in 1:N]
    add_data!(elm, samples, targets, sample_weights, batch_size = batch_size)
end

function add_data!(elm::T₁,
                   samples::T₂,
                   targets::T₃,
                   sample_weights::T₄;
                   batch_size::Int = 1000) where {T₁ <: ELM,
                                                  T₂ <: AbstractMatrix,
                                                  T₃ <: AbstractVector,
                                                  T₄ <: AbstractVector}
    add_data!(elm, samples, reshape(targets, 1, :), sample_weights,
              batch_size = batch_size)
end

function add_data!(elm::T₁,
                   samples::T₂,
                   targets::T₃;
                   batch_size::Int = 1000) where {T₁ <: ELM,
                                                  T₂ <: AbstractMatrix,
                                                  T₃ <: AbstractVector}
    add_data!(elm, samples, reshape(targets, 1, :), batch_size = batch_size)
end

function add_data!(elm::T₁,
                   sample::T₂,
                   targets::T₃,
                   sample_weights::T₄;
                   batch_size::Int = 1000) where {T₁ <: ELM,
                                                  T₂ <: AbstractVector,
                                                  T₃ <: AbstractMatrix,
                                                  T₄ <: AbstractVector}
    add_data!(elm, reshape(sample, 1, :), targets, sample_weights,
              batch_size = batch_size)
end

function add_data!(elm::T₁,
                   sample::T₂,
                   targets::T₃;
                   batch_size::Int = 1000) where {T₁ <: ELM,
                                                  T₂ <: AbstractVector,
                                                  T₃ <: AbstractMatrix}
    add_data!(elm, reshape(sample, 1, :), targets, batch_size = batch_size)
end


function add_data!(elm::T₁,
                   sample::T₂,
                   targets::T₃,
                   sample_weights::T₄;
                   batch_size::Int = 1000) where {T₁ <: ELM,
                                                  T₂ <: AbstractVector,
                                                  T₃ <: AbstractVector,
                                                  T₄ <: AbstractVector}
    add_data!(elm, reshape(sample, 1, :), reshape(targets, 1, :),
              sample_weights, batch_size = batch_size)
end

function add_data!(elm::T₁,
                   sample::T₂,
                   targets::T₃;
                   batch_size::Int = 1000) where {T₁ <: ELM,
                                                  T₂ <: AbstractVector,
                                                  T₃ <: AbstractVector}
    add_data!(elm, reshape(sample, 1, :). reshape(targets, 1, :),
              batch_size = batch_size)
end

function add_batch!(elm::T₁,
                    samples::T₂,
                    targets::T₃,
                    sample_weights::T₄) where {T₁ <: ELM,
                                               T₂ <: AbstractMatrix,
                                               T₃ <: AbstractMatrix,
                                               T₄ <: AbstractVector}
    X = samples
    T = targets
    Ψ = LinearAlgebra.Diagonal(sample_weights)

    # Get H, the nonlinear representation of X.
    H = project(elm.hidden_layer, X)

    # Adjust the weights of the samples and targets to compensate for any
    # imbalance in the frequency with which categories appear in the dataset.
    H = H * Ψ
    T = T * Ψ

    # Increment `elm.HH` and `elm.TH`, two covariance matrices which preserve
    # the intermediate state of the ELM, before it is solved.
    elm.hidden_layer.HH .+= (H * H')
    elm.hidden_layer.TH .+= (T * H')

    elm
end

function clear!(elm::T) where {T <: ELM}
    elm.hidden_layer.HH = nothing
    elm.hidden_layer.TH = nothing
    elm.hidden_layer.initialized = false
    elm
end

# Find the optimal output weights for the model by multiplying `elm.TH` by the
# pseudoinverse of `elm.HH`, and return the solved model as an `SLFN` object.
function solve(elm::T) where {T <: ELM}
    D = elm.n_features
    Q = elm.n_outputs
    hidden_layer = make_immutable(elm.hidden_layer)
    output_weights = elm.hidden_layer.TH * LinearAlgebra.pinv(elm.hidden_layer.HH)
    SLFN(D, Q, hidden_layer, output_weights)
end
