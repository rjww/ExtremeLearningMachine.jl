struct ELM{T <: Number}
    n_features::Int
    n_outputs::Int
    hidden_layer::MutableHiddenLayer

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

function add_neurons!(elm::ELM,
                      n_neurons::Int,
                      activation_function::F) where {F <: ActivationFunction}
    @assert begin
        isnothing(elm.hidden_layer.HH) &&
        isnothing(elm.hidden_layer.TH)
    end "Call clear! on ELM before adding neurons."

    T = getparams(elm.hidden_layer)
    neurons = Neurons(T, n_neurons, elm.n_features, activation_function)
    elm.hidden_layer.n_neurons += n_neurons
    push!(elm.hidden_layer.neurons, neurons)
    elm
end

function add_data!(elm::ELM,
                   samples::T1,
                   targets::T2,
                   sample_weights::T3;
                   batch_size::Int = 1000) where {T1 <: AbstractMatrix,
                                                  T2 <: AbstractMatrix,
                                                  T3 <: AbstractVector}
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

function add_data!(elm::ELM,
                   samples::T1,
                   targets::T2;
                   batch_size::Int = 1000) where {T1 <: AbstractMatrix,
                                                  T2 <: AbstractMatrix}
    N = last(size(samples))
    sample_weights = [1.0 for n in 1:N]
    add_data!(elm, samples, targets, sample_weights, batch_size = batch_size)
end

function add_data!(elm::ELM,
                   samples::T1,
                   targets::T2,
                   sample_weights::T3;
                   batch_size::Int = 1000) where {T1 <: AbstractMatrix,
                                                  T2 <: AbstractVector,
                                                  T3 <: AbstractVector}
    add_data!(elm, samples, reshape(targets, 1, :), sample_weights,
              batch_size = batch_size)
end

function add_data!(elm::ELM,
                   samples::T1,
                   targets::T2;
                   batch_size::Int = 1000) where {T1 <: AbstractMatrix,
                                                  T2 <: AbstractVector}
    add_data!(elm, samples, reshape(targets, 1, :), batch_size = batch_size)
end

function add_data!(elm::ELM,
                   sample::T1,
                   targets::T2,
                   sample_weights::T3;
                   batch_size::Int = 1000) where {T1 <: AbstractVector,
                                                  T2 <: AbstractMatrix,
                                                  T3 <: AbstractVector}
    add_data!(elm, reshape(sample, 1, :), targets, sample_weights,
              batch_size = batch_size)
end

function add_data!(elm::ELM,
                   sample::T1,
                   targets::T2;
                   batch_size::Int = 1000) where {T1 <: AbstractVector,
                                                  T2 <: AbstractMatrix}
    add_data!(elm, reshape(sample, 1, :), targets, batch_size = batch_size)
end


function add_data!(elm::ELM,
                   sample::T1,
                   targets::T2,
                   sample_weights::T3;
                   batch_size::Int = 1000) where {T1 <: AbstractVector,
                                                  T2 <: AbstractVector,
                                                  T3 <: AbstractVector}
    add_data!(elm, reshape(sample, 1, :), reshape(targets, 1, :),
              sample_weights, batch_size = batch_size)
end

function add_data!(elm::ELM,
                   sample::T1,
                   targets::T2;
                   batch_size::Int = 1000) where {T1 <: AbstractVector,
                                                  T2 <: AbstractVector}
    add_data!(elm, reshape(sample, 1, :). reshape(targets, 1, :),
              batch_size = batch_size)
end

function add_batch!(elm::ELM,
                    samples::T1,
                    targets::T2,
                    sample_weights::T3) where {T1 <: AbstractMatrix,
                                               T2 <: AbstractMatrix,
                                               T3 <: AbstractVector}
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

function clear!(elm::ELM)
    elm.hidden_layer.HH = nothing
    elm.hidden_layer.TH = nothing
    elm.hidden_layer.initialized = false
    elm
end

# Get H, the nonlinear representation of the sample array X. First, we project
# X onto R random hyperplanes (where R is the number of neurons in the hidden
# layer) by taking the product of the input weights W and X. Then, we apply
# some nonlinear operation f (the activation function for each neuron) to each
# element of the projection.
function project(hidden_layer::HiddenLayer,
                 samples::T) where {T <: AbstractMatrix}
    X = samples
    L = hidden_layer.n_neurons
    N = last(size(X))

    H = ones(getparams(hidden_layer), L + 1, N)
    i₀ = 1

    for (i, neurons) in enumerate(hidden_layer.neurons)
        L₀ = neurons.n_neurons
        f = neurons.activation_function
        W = neurons.weights

        i₁ = i₀ + L₀ - 1

        H[i₀:i₁,:] .= f.(W * X)

        i₀ = i₁ + 1
    end

    H
end

# Find the optimal output weights for the model by multiplying `elm.TH` by the
# pseudoinverse of `elm.HH`, and return the solved model as an `SLFN` object.
function solve(elm::ELM)
    D = elm.n_features
    Q = elm.n_outputs
    hidden_layer = make_immutable(elm.hidden_layer)
    output_weights = elm.hidden_layer.TH * LinearAlgebra.pinv(elm.hidden_layer.HH)
    SLFN(D, Q, hidden_layer, output_weights)
end
