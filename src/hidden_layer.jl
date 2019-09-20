abstract type HiddenLayer{T <: Number} end

mutable struct MutableHiddenLayer{T} <: HiddenLayer{T}
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

struct ImmutableHiddenLayer{T} <: HiddenLayer{T}
    n_neurons::Int
    neurons::Vector{Neurons}
end

getparams(hidden_layer::HiddenLayer{T}) where {T} = T

function is_initialized(hidden_layer::MutableHiddenLayer)
    hidden_layer.initialized
end

function initialize!(hidden_layer::MutableHiddenLayer,
                     n_outputs::Int)
    T = getparams(hidden_layer)
    L = hidden_layer.n_neurons
    Q = n_outputs

    hidden_layer.HH = zeros(T, L + 1, L + 1)
    hidden_layer.TH = zeros(T, Q, L + 1)
    hidden_layer.initialized = true

    hidden_layer
end

function make_immutable(hidden_layer::MutableHiddenLayer)
    T = getparams(hidden_layer)
    n_neurons = hidden_layer.n_neurons
    neurons = deepcopy(hidden_layer.neurons)
    ImmutableHiddenLayer{T}(n_neurons, neurons)
end
