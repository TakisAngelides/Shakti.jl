CI = get(ENV, "CI", nothing) == "true" || get(ENV, "GITHUB_TOKEN", nothing) !== nothing
using Documenter
ENV["JULIA_DEBUG"] = "Documenter"

using Shakti

PAGES = [
    "index.md",
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
    checkdocs = :none,
    warnonly = true,
)

deploydocs(
    repo = "github.com/TakisAngelides/Shakti.jl",
)
