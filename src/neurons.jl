struct Neurons{T₁ <: Number,
               T₂ <: ActivationFunction}
    n_neurons::Int
    weights::Matrix{T₁}
    activation_function::T₂

    function Neurons(::Type{T₁},
                     n_neurons::Int,
                     n_features::Int,
                     activation_function::T₂) where {T₁ <: Number,
                                                     T₂ <: ActivationFunction}
        weights = gaussian_projection_matrix(T₁, n_neurons, n_features)
        new{T₁,T₂}(n_neurons, weights, activation_function)
    end

    function Neurons(n_neurons::Int,
                     weights::AbstractMatrix{T₁},
                     activation_function::T₂) where {T₁ <: Number,
                                                     T₂ <: ActivationFunction}
        new{T₁,T₂}(n_neurons, weights, activation_function)
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
