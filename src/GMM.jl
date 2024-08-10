module GMM

export GMMSolution, gmm, gmmOPT
export Regression, MvRegression
export regOLS, multiOLS, regIV, multiIV, report
export exact, forwarddiff
export preset, nw, hh, white

include("dependencies.jl") # Package dependencies
include("main_gmm.jl") # Core functions for general GMM problems
include("linear_univariate.jl") # Functions for single-equation linear regression
include("linear_multivariate.jl") # Functions for multi-equation linear regression
include("moment_derivatives.jl") # Algorithms for computing first difference of moment functions
include("spectral_estimators.jl") # Algorithms for estimating spectral density of sample moments

end
