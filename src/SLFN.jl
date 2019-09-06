struct SLFN{T <: Number}
    n_features::Int
    n_outputs::Int
    hidden_layer::HiddenLayer
    output_weights::Matrix{T}
end

function predict(model::SLFN,
                 samples::T) where {T <: AbstractMatrix}
    X = samples
    H = project(model.hidden_layer, X)
    Wₒ = model.output_weights
    Y = Wₒ * H
    typeof(Y) <: AbstractMatrix && first(size(Y)) == 1 ? vec(Y) : Y
end

function predict(model::SLFN,
                 sample::T) where {T <: AbstractVector}
    predict(model, reshape(sample, 1, :))
end
