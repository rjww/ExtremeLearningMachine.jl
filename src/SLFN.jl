struct SLFN{T <: Number}
    n_features::Int
    n_outputs::Int
    hidden_layer::HiddenLayer
    output_weights::Matrix{T}
end

function predict(model::SLFN,
                 samples::AbstractArray{T}) where {T <: Number}
    X = samples
    H = project(model.hidden_layer, X)
    Wₒ = model.output_weights
    Y = Wₒ * H
end
