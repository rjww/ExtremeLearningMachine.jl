abstract type ActivationFunction end

struct Linear <: ActivationFunction end
(::Linear)(x::T) where T <: Real = x

struct ReLU <: ActivationFunction end
(::ReLU)(x::T) where T <: Real = max(zero(x), x)

struct Sigmoid <: ActivationFunction end
(::Sigmoid)(x::T) where T <: Real = 1 / (1 + exp(-x))

struct Square <: ActivationFunction end
(::Square)(x::T) where T <: Real = x^2

struct Tanh <: ActivationFunction end
(::Tanh)(x::T) where T <: Real = tanh(x)
