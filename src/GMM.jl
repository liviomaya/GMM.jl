module GMM

export gmm, GMMSolution
export Regression, MvRegression
export regOLS, multiOLS, regIV, multiIV, report
export exact, forwarddiff, nw, hh, white, preset
export BFGS, Newton

include("header.jl")
include("main_gmm.jl")
include("options.jl")
include("linear_univariate.jl")
include("linear_multivariate.jl")

end
