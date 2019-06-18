struct HiddenLayer{F <: ActivationFunction}
    n_neurons::Int
    activation_function::F
    input_weights::Matrix{Float64}

    function HiddenLayer(n_neurons::Int,
                         activation_function::F,
                         input_weights::Matrix{Float64}) where F <: ActivationFunction
        new{F}(n_neurons, activation_function, input_weights)
    end

    function HiddenLayer(n_neurons::Int,
                         n_features::Int,
                         activation_function::F) where F <: ActivationFunction
        input_weights = gaussian_projection_matrix(n_neurons, n_features)
        new{F}(n_neurons, activation_function, input_weights)
    end
end

function Base.copy(hl::HiddenLayer)
    fields = [deepcopy(getfield(hl, k)) for k ∈ fieldnames(HiddenLayer)]
    HiddenLayer(fields...)
end

function gaussian_projection_matrix(n_neurons::Int, n_features::Int)
    randn(n_neurons, n_features) ./ sqrt(n_features)
end
