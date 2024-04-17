using Base
using LinearAlgebra
using DelimitedFiles

function read_pc(filename::String)
    data, _ = readdlm(filename, Int, ' ')  # ' ' specifies that the delimiter is whitespace
    return data
end

deg_bit = 4
deg_check = 5
readdir = "/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/rgg_code"
readpath = joinpath(readdir, "hclassical_n{n}_m{m}_degbit{}")
savepath = os.path.join(savedir, f'hclassical_n{n}_m{m}_degbit{deg_bit}_degcheck{deg_check}_r{r}_seed{seed}.txt')
data = read_pc(savepath)
