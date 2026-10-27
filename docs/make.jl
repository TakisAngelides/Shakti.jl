CI = get(ENV, "CI", nothing) == "true" || get(ENV, "GITHUB_TOKEN", nothing) !== nothing
using CairoMakie, Documenter, Literate
using DocumenterCitations
ENV["JULIA_DEBUG"] = "Documenter"

using Shakti

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style = :authoryear)

# Literate pages run live at build time. The real-data examples (Helheim, Drang Drung, Thwaites,
# Greenland, Antarctica) are plain markdown with figures made by make_figures.jl.
Literate.markdown(joinpath(@__DIR__, "src", "tutorial.jl"), joinpath(@__DIR__, "src"); config = Dict("credit" => false))

literate_examples = ["SeasonalMeltInput", "ParabolicScheme", "SUHMO", "GapCoupling"]
for example in literate_examples
    Literate.markdown(joinpath(@__DIR__, "src", "examples", "$example.jl"), joinpath(@__DIR__, "src", "examples");
                      config = Dict("credit" => false))
end

example_pages = [
    "examples/Helheim.md",
    "examples/DrangDrung.md",
    "examples/Thwaites.md",
    "examples/Greenland.md",
    "examples/Antarctica.md",
]

more_pages = [["examples/$example.md" for example in literate_examples]..., "performance.md"]

PAGES = [
    "index.md",
    "Model" => ["equations.md", "discretization.md"],
    "tutorial.md",
    "Examples" => example_pages,
    "More" => more_pages,
    "References" => ["API_public.md", "references.md"],
]

makedocs(
    modules = [Shakti],
    format = Documenter.HTML(
        prettyurls = CI,
        collapselevel = 2,
        size_threshold = 1_000_000,
        assets = ["assets/tables.css"],
    ),
    sitename = "Shakti.jl",
    authors = "Takis Angelides",
    pages = PAGES,
    doctest = CI,
    draft = false,
    plugins = [bib],
    checkdocs = :exports,
    checkdocs_ignored_modules = [Shakti.Data],   # ParallelStencil's generated submodule, not ours
    warnonly = true,
)

deploydocs(
    repo = "github.com/TakisAngelides/Shakti.jl",
)
