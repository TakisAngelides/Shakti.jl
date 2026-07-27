CI = get(ENV, "CI", nothing) == "true" || get(ENV, "GITHUB_TOKEN", nothing) !== nothing
using Documenter, Literate
ENV["JULIA_DEBUG"] = "Documenter"

using Shakti

for example in ("SeasonalMeltInput", "MaskVariants", "Helheim")
    Literate.markdown(
        joinpath(@__DIR__, "src", "examples", "$example.jl"),
        joinpath(@__DIR__, "src", "examples");
        config = Dict("credit" => false)
    )
end

example_pages = [
    "examples/SeasonalMeltInput.md",
    "examples/MaskVariants.md",
    "examples/Helheim.md",
]

PAGES = [
    "index.md",
    "Examples" => example_pages,
    "API.md",
]

makedocs(
    modules = [Shakti],
    format = Documenter.HTML(
        prettyurls = CI,
        collapselevel = 2,
    ),
    sitename = "Shakti.jl",
    authors = "Takis Angelides",
    pages = PAGES,
    doctest = CI,
    draft = false,
    checkdocs = :exports,
    warnonly = true,
)

deploydocs(
    repo = "github.com/TakisAngelides/Shakti.jl",
)
