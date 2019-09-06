struct ELM{T <: Number}
    n_features::Int
    n_outputs::Int
    hidden_layer::HiddenLayer
    HH::Matrix{T}
    TH::Matrix{T}

    function ELM{T}(n_features::Int,
                    n_outputs::Int,
                    n_neurons::Int,
                    activation_function::ActivationFunction) where {T <: Number}
        hidden_layer = HiddenLayer(T, n_neurons, n_features, activation_function)
        HH = zeros(T, n_neurons + 1, n_neurons + 1)
        TH = zeros(T, n_outputs, n_neurons + 1)
        new(n_features, n_outputs, hidden_layer, HH, TH)
    end

    function ELM(n_features::Int,
                 n_outputs::Int,
                 n_neurons::Int,
                 activation_function::ActivationFunction)
        ELM{Float64}(n_features, n_outputs, n_neurons, activation_function)
    end
end

function add_data!(elm::ELM,
                   samples::T1,
                   targets::T2,
                   sample_weights::T3;
                   batch_size::Int = 1000) where {T1 <: AbstractMatrix,
                                                  T2 <: AbstractMatrix,
                                                  T3 <: AbstractVector}
    X = samples
    T = targets
    ψ = sample_weights

    dₓ, Nₓ = size(X)
    qₜ, Nₜ = size(T)

    @assert dₓ == elm.n_features "Input dimensionality mismatch."
    @assert qₜ == elm.n_outputs "Output dimensionality mismatch."
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
    elm.HH .+= (H * H')
    elm.TH .+= (T * H')

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
    f = hidden_layer.activation_function
    W = hidden_layer.input_weights
    H = ones(eltype(W), first(size(W)) + 1, last(size(X)))
    H[1:end-1,:] .= f.(W * X)
    H
end

# Find the optimal output weights for the model by multiplying `elm.TH` by the
# pseudoinverse of `elm.HH`, and return the solved model as an `SLFN` object.
function solve(elm::ELM)
    hidden_layer = copy(elm.hidden_layer)
    output_weights = elm.TH * LinearAlgebra.pinv(elm.HH)
    SLFN(elm.n_features, elm.n_outputs, hidden_layer, output_weights)
end
