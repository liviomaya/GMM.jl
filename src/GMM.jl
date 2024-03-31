module GMM

export gmm, GMMSolution
export reg, reg_table, mv_reg
export exact, finite_diff
export newey_west, hansen_hodrick, white, preset
export BFGS, Newton

include("f0_header.jl")
include("f1_gmm.jl")
include("f2_options.jl")
include("f3_reg.jl")

end # module
