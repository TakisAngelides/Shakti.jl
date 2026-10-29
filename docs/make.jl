CI = get(ENV, "CI", nothing) == "true" || get(ENV, "GITHUB_TOKEN", nothing) !== nothing
using Documenter, Literate
using DocumenterCitations
ENV["JULIA_DEBUG"] = "Documenter"

using Shakti

# SHAKTI_DOCS_PREVIEW=1 builds only the model pages, without running the Literate examples or @example blocks, for live
# editing (see docs/preview.jl). The full build is unchanged.
PREVIEW = get(ENV, "SHAKTI_DOCS_PREVIEW", "") == "1"
PREVIEW || using CairoMakie   # slow to load, and only the executed examples need it

# Documenter processes every page in the source folder, listed or not, so the preview builds from a
# folder holding links to just the model pages.
SRC = joinpath(@__DIR__, "src")
if PREVIEW
    SRC = joinpath(@__DIR__, "build_preview_src")
    rm(SRC; recursive = true, force = true)
    mkpath(SRC)
    for f in ["index.md", "equations.md", "discretization.md", "numerical_details.md", "references.md", "refs.bib", "assets", "figures"]
        symlink(joinpath(@__DIR__, "src", f), joinpath(SRC, f))
    end
end

bib = CitationBibliography(joinpath(SRC, "refs.bib"); style = :authoryear)

# Literate pages run live at build time. The real-data examples (Helheim, Drang Drung, Thwaites,
# Greenland, Antarctica) are plain markdown with figures made by make_figures.jl.
literate_examples = ["SeasonalMeltInput", "ParabolicScheme", "SUHMO", "GapCoupling"]
if !PREVIEW
    Literate.markdown(joinpath(@__DIR__, "src", "tutorial.jl"), joinpath(@__DIR__, "src"); config = Dict("credit" => false))
    for example in literate_examples
        Literate.markdown(joinpath(@__DIR__, "src", "examples", "$example.jl"), joinpath(@__DIR__, "src", "examples");
                          config = Dict("credit" => false))
    end
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
    "Model" => ["equations.md", "discretization.md", "numerical_details.md"],
    "tutorial.md",
    "Examples" => example_pages,
    "More" => more_pages,
    "References" => ["API_public.md", "references.md"],
]
PREVIEW && (PAGES = ["index.md", "Model" => ["equations.md", "discretization.md", "numerical_details.md"], "references.md"])

makedocs(
    source = SRC,
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
    draft = PREVIEW,   # the preview does not run @example blocks
    plugins = [bib],
    checkdocs = PREVIEW ? :none : :exports,
    checkdocs_ignored_modules = [Shakti.Data],   # ParallelStencil's generated submodule, not ours
    warnonly = true,
)

PREVIEW || deploydocs(
    repo = "github.com/TakisAngelides/Shakti.jl",
)
