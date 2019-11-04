abstract type HiddenLayer{T <: Number} end

mutable struct MutableHiddenLayer{T <: Number} <: HiddenLayer{T}
    n_neurons::Int
    neurons::Vector{Neurons}
    HH::Union{Matrix{T}, Nothing}
    TH::Union{Matrix{T}, Nothing}
    initialized::Bool

    function MutableHiddenLayer(::Type{T}) where {T <: Number}
        n_neurons = 0
        neurons = Vector{Neurons}()
        HH = nothing
        TH = nothing
        initialized = false
        new{T}(n_neurons, neurons, HH, TH, initialized)
    end
end

struct ImmutableHiddenLayer{T <: Number} <: HiddenLayer{T}
    n_neurons::Int
    neurons::Vector{Neurons}
end

function is_initialized(hidden_layer::T) where {T <: MutableHiddenLayer}
    hidden_layer.initialized
end

function initialize!(hidden_layer::T₁,
                     n_outputs::Int) where {T₁ <: MutableHiddenLayer}

    param(::MutableHiddenLayer{T}) where {T} = T

    L = hidden_layer.n_neurons
    Q = n_outputs

    hidden_layer.HH = zeros(param(hidden_layer), L + 1, L + 1)
    hidden_layer.TH = zeros(param(hidden_layer), Q, L + 1)
    hidden_layer.initialized = true

    hidden_layer
end

function make_immutable(hidden_layer::MutableHiddenLayer)
    param(::HiddenLayer{T}) where {T} = T

    n_neurons = hidden_layer.n_neurons
    neurons = deepcopy(hidden_layer.neurons)
    ImmutableHiddenLayer{param(hidden_layer)}(n_neurons, neurons)
end

# Get H, the nonlinear representation of the sample array X. First, we project
# X onto R random hyperplanes (where R is the number of neurons in the hidden
# layer) by taking the product of the input weights W and X. Then, we apply
# some nonlinear operation f (the activation function for each neuron) to each
# element of the projection.
function project(hidden_layer::T₁,
                 samples::T₂) where {T₁ <: HiddenLayer,
                                     T₂ <: AbstractMatrix}
    param(::HiddenLayer{T}) where {T} = T

    X = samples
    L = hidden_layer.n_neurons
    N = last(size(X))

    H = ones(param(hidden_layer), L + 1, N)
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
