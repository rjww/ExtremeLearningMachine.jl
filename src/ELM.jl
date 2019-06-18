struct ELM
    n_features::Int
    n_outputs::Int
    hidden_layer::HiddenLayer
    HH::Matrix{Float64}
    TH::Matrix{Float64}

    function ELM(n_features::Int,
                 n_outputs::Int,
                 n_neurons::Int,
                 activation_function::ActivationFunction)
        hidden_layer = HiddenLayer(n_neurons, n_features, activation_function)
        HH = zeros(n_neurons, n_neurons)
        TH = zeros(n_outputs, n_neurons)
        new(n_features, n_outputs, hidden_layer, HH, TH)
    end
end

function add_data!(elm::ELM,
                   samples::AbstractArray{T1},
                   targets::AbstractArray{T2},
                   sample_weights::AbstractVector{T3};
                   batch_size::Int = 1000) where {T1, T2, T3 <: Real}
    X = samples
    T = targets
    ψ = sample_weights

    dₓ, Nₓ = size(X)
    qₜ, Nₜ = size(T)

    @assert dₓ == elm.n_features "Input dimensionality mismatch."
    @assert qₜ == elm.n_targets "Output dimensionality mismatch."
    @assert Nₓ == Nₜ == length(ψ) "Sample count mismatch."

    for batch in partition_range(1:N, batch_size)
        X₀ = X[:,batch]
        T₀ = T[:,batch]
        ψ₀ = ψ[batch]
        add_batch!(elm, X₀, T₀, ψ₀)
    end

    elm
end

function add_data!(elm::ELM,
                   samples::AbstractArray{T1},
                   targets::AbstractArray{T2};
                   batch_size::Int = 1000) where {T1, T2 <: Real}
    N = last(size(samples))
    sample_weights = [1.0 for n in 1:N]
    add_data!(elm, samples, targets, sample_weights, batch_size=batch_size)
end

function add_batch!(elm::ELM,
                    samples::AbstractArray{T1},
                    targets::AbstractArray{T2},
                    sample_weights::AbstractVector{T3}) where {T1, T2, T3 <: Real}
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
                 samples::AbstractArray{T}) where T <: Real
    X = samples
    f = hidden_layer.activation_function
    W = hidden_layer.weights
    H = f.(W * X)
end

# Find the optimal output weights for the model by multiplying `elm.TH` by the
# pseudoinverse of `elm.HH`, and return the solved model as an `SLFN` object.
function solve(elm::ELM)
    hidden_layer = copy(elm.hidden_layer)
    output_weights = elm.TH * LinearAlgebra.pinv(elm.HH)
    SLFN(elm.n_features, elm.n_outputs, hidden_layer, output_weights)
end
