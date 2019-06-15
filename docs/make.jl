using Documenter, ExtremeLearningMachine

makedocs(;
    modules=[ExtremeLearningMachine],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/rjww/ExtremeLearningMachine.jl/blob/{commit}{path}#L{line}",
    sitename="ExtremeLearningMachine.jl",
    authors="Robert Woods",
    assets=String[],
)

deploydocs(;
    repo="github.com/rjww/ExtremeLearningMachine.jl",
)
