using Pkg
Pkg.activate("docs")
Pkg.develop(PackageSpec(path=pwd()))
Pkg.instantiate()

using MRIeddyCurrentOptimization
using Documenter
using Literate
using Plots # to not capture precompilation output

# HTML Plotting Functionality
struct HTMLPlot
    p # :: Plots.Plot
end
const ROOT_DIR = joinpath(@__DIR__, "build")
const PLOT_DIR = joinpath(ROOT_DIR, "plots")
function Base.show(io::IO, ::MIME"text/html", p::HTMLPlot)
    mkpath(PLOT_DIR)
    path = joinpath(PLOT_DIR, string(UInt32(floor(rand()*1e9)), ".html"))
    Plots.savefig(p.p, path)
    # if get(ENV, "CI", "false") == "true" # for prettyurl
    #     print(io, "<object type=\"text/html\" data=\"../$(relpath(path, ROOT_DIR))\" style=\"width:100%;height:425px;\"></object>")
    # else
        print(io, "<object type=\"text/html\" data=\"$(relpath(path, ROOT_DIR))\" style=\"width:100%;height:425px;\"></object>")
    # end
end

# Notebook hack to display inline math correctly
function notebook_filter(str)
    re = r"(?<!`)``(?!`)"  # Two backquotes not preceded by nor followed by another
    replace(str, re => "\$")
end

# Literate
OUTPUT = joinpath(@__DIR__, "src/build_literate")

files = [
    "index.jl",
]

for file in files
    file_path = joinpath(@__DIR__, "src/", file)
    Literate.markdown(file_path, OUTPUT)
    Literate.notebook(file_path, OUTPUT, preprocess=notebook_filter; execute=false)
    Literate.script(  file_path, OUTPUT)
end

src = joinpath(@__DIR__, "src/build_literate/index.md")
tar = joinpath(@__DIR__, "src/index.md")
run(`mv $src $tar`)

DocMeta.setdocmeta!(MRIeddyCurrentOptimization, :DocTestSetup, :(using MRIeddyCurrentOptimization); recursive=true)

makedocs(;
    doctest = true,
    modules=[MRIeddyCurrentOptimization],
    authors="Jakob Asslaender <jakob.asslaender@nyumc.org> and Sebastian Flassbeck <sebastian.flassbeck@nyumc.org>",
    sitename="MRIeddyCurrentOptimization.jl",
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", "false") == "true"),
    pages=[
        "Home" => "index.md",
        # "Home" => "build_literate/tutorial.md",
        "API" => "api.md",
    ],
)

# Set dark theme as default independent of the OS's settings
run(`sed -i'.old' 's/var darkPreference = false/var darkPreference = true/g' docs/build/assets/themeswap.js`)

deploydocs(;
    repo="github.com/JakobAsslaender/MRIeddyCurrentOptimization.jl",
)
