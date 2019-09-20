module ExtremeLearningMachine

import LinearAlgebra

include("util.jl")
include("activation_functions.jl")
include("neurons.jl")
include("hidden_layer.jl")
include("slfn.jl")
include("elm.jl")

export ELM,
       Linear, ReLU, Sigmoid, Square, Tanh,
       add_neurons!, add_data!, clear!, solve, predict

end # module
