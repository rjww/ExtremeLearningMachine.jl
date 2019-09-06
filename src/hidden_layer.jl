struct HiddenLayer{F <: ActivationFunction, T <: Number}
    n_neurons::Int
    activation_function::F
    input_weights::Matrix{T}

    function HiddenLayer(n_neurons::Int,
                         activation_function::F,
                         input_weights::Matrix{T}) where {F <: ActivationFunction, T <: Number}
        new{F,T}(n_neurons, activation_function, input_weights)
    end

    function HiddenLayer(::Type{T},
                         n_neurons::Int,
                         n_features::Int,
                         activation_function::F) where {F <: ActivationFunction, T <: Number}
        input_weights = gaussian_projection_matrix(T, n_neurons, n_features)
        new{F,T}(n_neurons, activation_function, input_weights)
    end
end

function Base.copy(hl::HiddenLayer)
    fields = [deepcopy(getfield(hl, k)) for k ∈ fieldnames(HiddenLayer)]
    HiddenLayer(fields...)
end

function gaussian_projection_matrix(::Type{T}, n_neurons::Int, n_features::Int) where {T}
    randn(T, n_neurons, n_features) ./ sqrt(n_features)
end
