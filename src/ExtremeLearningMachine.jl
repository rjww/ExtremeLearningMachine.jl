module ExtremeLearningMachine

import LinearAlgebra

include("util.jl")
include("activation_functions.jl")
include("hidden_layer.jl")
include("slfn.jl")
include("elm.jl")

export ELM,
       Linear, ReLU, Sigmoid, Square, Tanh,
       add_data!, solve, predict

end # module
