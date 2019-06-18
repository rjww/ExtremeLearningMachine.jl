struct SLFN
    n_features::Int
    n_outputs::Int
    hidden_layer::HiddenLayer
    output_weights::Matrix{Float64}
end

function predict(model::SLFN,
                 samples::AbstractArray{T}) where T <: Real
    X = samples
    H = project(model.hidden_layer, X)
    Wₒ = model.output_weights
    Y = Wₒ * H
end
