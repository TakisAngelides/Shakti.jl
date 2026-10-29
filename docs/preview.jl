# Live preview of the model pages: rebuilds and reloads the browser on every save.
#   julia --project=docs docs/preview.jl
# LiveServer is not a docs dependency; install it once in the global environment:
#   julia -e 'using Pkg; Pkg.add("LiveServer")'
ENV["SHAKTI_DOCS_PREVIEW"] = "1"
using LiveServer
servedocs(; foldername = @__DIR__, skip_dirs = [joinpath(@__DIR__, "src", "examples")])
