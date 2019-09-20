struct Neurons{F <: ActivationFunction, T <: Number}
    n_neurons::Int
    activation_function::F
    weights::Matrix{T}

    function Neurons(::Type{T},
                     n_neurons::Int,
                     n_features::Int,
                     activation_function::F) where {F <: ActivationFunction,
                                                    T <: Number}
        weights = gaussian_projection_matrix(T, n_neurons, n_features)
        new{F,T}(n_neurons, activation_function, weights)
    end

    function Neurons(n_neurons::Int,
                     activation_function::F,
                     weights::AbstractMatrix{T}) where {F <: ActivationFunction,
                                                        T <: Number}
        new{F,T}(n_neurons, activation_function, weights)
    end
end

function Base.copy(ns::Neurons)
    fields = [deepcopy(getfield(ns, k)) for k ∈ fieldnames(Neurons)]
    Neurons(fields...)
end

function gaussian_projection_matrix(::Type{T},
                                    n_neurons::Int,
                                    n_features::Int) where {T <: Number}
    randn(T, n_neurons, n_features) ./ sqrt(n_features)
end
