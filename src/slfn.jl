struct SLFN{T <: Number}
    n_features::Int
    n_outputs::Int
    hidden_layer::ImmutableHiddenLayer{T}
    output_weights::Matrix{T}
end

function predict(model::T₁,
                 samples::T₂) where {T₁ <: SLFN,
                                     T₂ <: AbstractMatrix}
    X = samples
    H = project(model.hidden_layer, X)
    Β = model.output_weights
    Y = Β * H
    typeof(Y) <: AbstractMatrix && first(size(Y)) == 1 ? vec(Y) : Y
end

function predict(model::T₁,
                 sample::T₂) where {T₁ <: SLFN,
                                    T₂ <: AbstractVector}
    predict(model, reshape(sample, 1, :))
end
