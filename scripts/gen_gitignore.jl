#=
  @ author: ChenyuBao <chenyu.bao@outlook.com>
  @ date: 2025-12-07 20:17:22
  @ license: MIT
  @ language: Julia
  @ declaration: EtherArrays.jl is a wrapper of sized arrays on any device.
  @ description: /
 =#

const kSpecifiedFiles = ("Manifest.toml",)
const kIgnoredFiles = ("**.jld2",)
const kIgnoredFolders = (".vscode/**", "drafts/**")

function gen_gitignore()
    text = "# .gitignore files"
    text *= "\n\n"
    # specified files
    text *= "# specified files\n"
    for file in kSpecifiedFiles
        text *= file * "\n"
    end
    text *= "\n"
    # ignored files
    text *= "# ignored files\n"
    for file in kIgnoredFiles
        text *= file * "\n"
    end
    text *= "\n"
    # ignored folders
    text *= "# ignored folders\n"
    for folder in kIgnoredFolders
        text *= folder * "\n"
    end
    text *= "\n"
    open(".gitignore", "w") do io
        write(io, text)
    end
end

gen_gitignore()
