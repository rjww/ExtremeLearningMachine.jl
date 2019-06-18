module ExtremeLearningMachine

import LinearAlgebra

include("util.jl")
include("activation_functions.jl")
include("HiddenLayer.jl")
include("SLFN.jl")
include("ELM.jl")

export ELM,
       Linear, ReLU, Sigmoid, Square, Tanh,
       add_data!, solve, predict

end # module
