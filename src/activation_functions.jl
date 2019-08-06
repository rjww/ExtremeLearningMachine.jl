abstract type ActivationFunction end

struct Linear <: ActivationFunction end
(::Linear)(x::T) where {T <: Number} = x

struct ReLU <: ActivationFunction end
(::ReLU)(x::T) where {T <: Number} = max(zero(x), x)

struct Sigmoid <: ActivationFunction end
(::Sigmoid)(x::T) where {T <: Number} = 1 / (1 + exp(-x))

struct Square <: ActivationFunction end
(::Square)(x::T) where {T <: Number} = x^2

struct Tanh <: ActivationFunction end
(::Tanh)(x::T) where {T <: Number} = tanh(x)
