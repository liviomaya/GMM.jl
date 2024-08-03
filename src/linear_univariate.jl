"""
        Regression

Stores the solution to a single-equation regression model (see `regOLS` and `regIV` functions).

## Fields
- `gmm::GMMSolution`: solution to GMM problem on orthogonality conditions
- `nreg::Int`: number of regressors (includes intercept)
- `nobs::Int`: number of observations, or the sample size
- `intercept::Float`: equation intercept (`NaN` if no intercept)
- `coef::Vector{Float}`: estimated coefficients
- `stDev::Vector{Float}`: asymptotic standard deviation of coefficient estimates
- `tStat::Vector{Float}`: `coef` / `stDev`
- `resStDev::Float`: MLE estimate of residuals' standard deviation
- `fitted::Vector{Float}`: fitted values
- `res::Vector{Float}`: sample residuals
- `SSE::Float`: sum of squared residuals
- `R2::Float`: R-Squared
- `R2adj::Float`: adjusted R-Squared
- `logLLH::Float`: log-likelihood (assumes no residual serial correlation)
- `AIC::Float`: Akaike information criterion (assumes no residual serial correlation)
- `BIC::Float`: Bayesian information criterion (assumes no residual serial correlation)
"""
struct Regression
    gmm::GMMSolution
    nreg::Int64
    nobs::Int64
    intercept::Float64
    coef::Vector{Float64}
    stDev::Vector{Float64}
    tStat::Vector{Float64}
    resStDev::Float64
    fitted::Vector{Float64}
    res::Vector{Float64}
    SSE::Float64
    R2::Float64
    R2adj::Float64
    logLLH::Float64
    AIC::Float64
    BIC::Float64
end

function orthogonal(Y, X, Z, b)
    e = Y .- X * b
    mom = zeros(size(Z))
    for i in axes(Z, 1), j in axes(Z, 2)
        mom[i, j] = Z[i, j] * e[i]
    end
    return mom
end

function gaussian_loglikelihood(X::Vector{Float64}, mu::Float64, sigma::Float64)
    res = X .- mu
    T = length(res)
    logL = -(T / 2) * log(2 * pi)
    logL += -(T / 2) * log(sigma^2)
    logL += -(1 / (2 * sigma^2)) * sum(res .^ 2)
    return logL
end

function gmmiv_exact(Y, X, Z, spectral_model)
    T, nZ = size(Z)
    coef = (Z' * X) \ Z' * Y
    mom = zeros(nZ)
    Dmom = -(Z' * X) / T # - E(zₜxₜᵀ)
    momCov = zeros(nZ, nZ)
    f(b) = orthogonal(Y, X, Z, b)
    spectral = spectral_model(f, coef)
    coefCov = inv(Dmom) * spectral * inv(Dmom)'
    weight = diagm(ones(nZ))
    return GMMSolution(coef, mom, coefCov, momCov, Dmom, spectral, weight, T)
end

function gmmiv_over(Y, X, Z, weight, spectral_model)
    T, nmom = size(Z)
    coef = (X' * Z * weight * Z' * X) \ X' * Z * weight * Z' * Y
    mom = Z' * (Y .- X * coef) / T
    Dmom = -(Z' * X) / T # - E(zₜxₜᵀ)
    f(b) = orthogonal(Y, X, Z, b)
    spectral = spectral_model(f, coef)

    # H = pinv(Dmom' * weight * Dmom)
    H = inv(Dmom' * weight * Dmom)
    coefCov = H * Dmom' * weight * spectral * weight' * Dmom * H'
    momCov = (I(nmom) - Dmom * H * Dmom' * weight) * spectral * (I(nmom) - Dmom * H * Dmom' * weight)'
    return GMMSolution(coef, mom, coefCov, momCov, Dmom, spectral, weight, T)
end

function build_regression(gmmSol, y, X, intercept)
    stDev = sqrt.(diag(gmmSol.coefCov) / gmmSol.nobs)
    tStat = gmmSol.coef ./ stDev
    fitted = X * gmmSol.coef
    res = y .- fitted
    resStDev = sqrt(mean(res .^ 2))
    SSE = sum(res .^ 2)
    R2 = 1.0 - SSE / sum((y .- mean(y)) .^ 2)
    R2adj = 1 - (1 - R2) * (gmmSol.nobs - 1) / (gmmSol.nobs - gmmSol.npar)
    logLLH = gaussian_loglikelihood(res, 0.0, resStDev)

    # for AIC, check https://robjhyndman.com/hyndsight/lm_aic.html
    npar = gmmSol.npar + 1 # add error variance
    AIC = 2 * npar - 2 * logLLH
    BIC = log(gmmSol.nobs) * npar - 2 * logLLH

    return Regression(gmmSol,
        gmmSol.npar,
        gmmSol.nobs,
        intercept ? gmmSol.coef[1] : NaN,
        intercept ? gmmSol.coef[2:end] : gmmSol.coef,
        intercept ? stDev[2:end] : stDev,
        intercept ? tStat[2:end] : tStat,
        resStDev,
        fitted,
        res,
        SSE,
        R2,
        R2adj,
        logLLH,
        AIC,
        BIC)
end

function check_consistency(y, x, z, weight, intercept)

    if (length(y) == 0) | (length(x) == 0) | (length(z) == 0)
        throw(ArgumentError("One (of more) empty arrays."))
    end

    if !(size(y, 1) == size(x, 1) == size(z, 1))
        throw(ArgumentError("Sample arrays must have the same number of rows."))
    end

    if size(weight, 1) != size(weight, 2)
        throw(ArgumentError("Weighting matrix must be square."))
    end

    if size(weight, 2) != (size(z, 2) + intercept) * size(y, 2)
        throw(ArgumentError("Size of weighting matrix not consistent with number of moments."))
    end

end

function check_order(X, Z)
    if size(Z, 2) < size(X, 2)
        throw(ArgumentError("Order condition violated. Please, add instruments."))
    end
end

function check_rank(X, Z)
    if rank(Z' * X ./ size(X, 1); atol=1e-12) < size(X, 2)
        throw(ArgumentError("Rank condition violated. Columns of Z'X not linearly independent. (Tolerance = 1e-12.)"))
    end
end

"""
        reg = regOLS(y, x; <kwargs>)

Estimate `y = x β + e` by ordinary least squares (OLS). The output `reg` is a `Regression` object. 

### Arguments
- `y::Vector{Float}`: vector with explained variable sample
- `x::VecOrMat{Float}`: array with explanatory variable(s) sample (dim 1 = observations, dim 2 = variables)

### Keyword Arguments
- `intercept::Bool`: `true` to add intercept term as regressor  
- `spectral_model`: estimator of spectral density of moment sample (see documentation for `gmm` function). Choose between: 
    - `preset(S)` for a given `S::Array{Float, 3}` (dim 1: observations, dim 2: moments, dim 3: parameters)
    - `nw(k)` (Newey & West 1987), where `k::Int` is the number of lags 
    - `hh(k)` (Hansen & Hodrick 1980), where `k::Int` is the number of lags 
    - `white()` (White (1980), serially uncorrelated `f`) 
    Default = `white()` . 

"""
function regOLS(y::Array{Float64,1},
    x::J where {J<:VecOrMat{Float64}};
    intercept::Bool=true,
    spectral_model=white())

    return regIV(y, x, x;
        intercept=intercept,
        spectral_model=spectral_model)
end

"""
        reg = regIV(y, x, z; <kwargs>)

Estimate `y = x β + e` using instrumental variable (IV) `z`. The output `reg` is a `Regression` object. The order condition requires the number of instruments `z` to be as least as many as the number of explanatory variables `x`.

### Arguments
- `y::Vector{Float}`: vector with explained variable sample
- `x::VecOrMat{Float}`: array with explanatory variable(s) sample (dim 1 = observations, dim 2 = variables)
- `z::VecOrMat{Float}`: array with instrument(s) sample (dim 1 = observations, dim 2 = variables)

### Keyword Arguments
- `intercept::Bool`: `true` to add intercept term, both as regressor and as instrument
- `two_step::Bool`: `true` to estimate `β` two times, using efficient weights in the second run  
- `weight::Matrix`: weighting matrix for orthogonality conditions. Default = identity
- `spectral_model`: estimator of spectral density of moment sample (see documentation for `gmm` function). Choose between: 
    - `preset(S)` for a given `S::Array{Float, 3}` (dim 1: observations, dim 2: moments, dim 3: parameters)
    - `nw(k)` (Newey & West 1987), where `k::Int` is the number of lags 
    - `hh(k)` (Hansen & Hodrick 1980), where `k::Int` is the number of lags 
    - `white()` (White (1980), serially uncorrelated `f`) 
    Default = `white()` 
"""
function regIV(y::Array{Float64,1},
    x::J where {J<:VecOrMat{Float64}},
    z::H where {H<:VecOrMat{Float64}};
    intercept::Bool=true,
    two_step::Bool=false,
    weight::Matrix{Float64}=diagm(ones(size(z, 2) + intercept)),
    spectral_model=white())

    check_consistency(y, x, z, weight, intercept)

    # promote to matrix
    x_mat = promote_to_matrix(x)
    z_mat = promote_to_matrix(z)

    # check valid IV 
    check_order(x_mat, z_mat)
    check_rank(x_mat, z_mat)

    # add intercept
    X = intercept ? [ones(size(x_mat, 1)) x_mat] : x_mat
    Z = intercept ? [ones(size(z_mat, 1)) z_mat] : z_mat

    # compute solution
    if size(z_mat, 2) == size(x_mat, 2)
        gmmSol = gmmiv_exact(y, X, Z, spectral_model)

    elseif size(z_mat, 2) > size(x_mat, 2)
        use_weight = weight
        gmmSol = gmmiv_over(y, X, Z, use_weight, spectral_model)
        if two_step
            use_weight = inv(gmmSol.spectral)
            gmmSol = gmmiv_over(y, X, Z, use_weight, preset(gmmSol.spectral))
        end

    end

    return build_regression(gmmSol, y, X, intercept)
end

compute_pval(tstat) = 2 * (1 .- cdf(Normal(0, 1), abs(tstat)))

function stars(pval)
    pval < 0.01 && return "***"
    pval < 0.025 && return "**"
    pval < 0.05 && return "*"
    pval < 0.10 && return "."
    return ""
end

"""
        report(reg)

Print regression summary. Argument `reg` can be `Regression` (single-equation) or `MvRegression` (multiple-equation).
"""
function report(reg::Regression)

    coefTable = DataFrame(
        "variable" => String[],
        "coef" => Float64[],
        "std" => Float64[],
    )

    # table
    if !isnan(reg.intercept) # check for intercept
        push!(coefTable, ["(Intercept)" reg.intercept sqrt(reg.gmm.coefCov[1, 1] / reg.nobs)])
    end

    for p in eachindex(reg.coef)
        push!(coefTable, ["Variable $p" reg.coef[p] reg.stDev[p]])
    end
    coefTable.tstat = coefTable.coef ./ coefTable.std
    coefTable.pval = compute_pval.(coefTable.tstat)
    coefTable.stars = stars.(coefTable.pval)

    myround(x) = round(x, digits=3)

    println("")
    pretty_table(coefTable;
        title="Coefficients",
        tf=tf_compact,
        header=["", "Estim", "StDev", "tStat", "P(>|t|)", ""],
        formatters=ft_printf("%7.3f", 2:5)
    )
    println("Significance: 1% (***) 2.5% (**) 5% (*) 10% (.)")
    println("")
    println("R-Squared: $(myround(reg.R2)), Adjusted R-Squared: $(myround(reg.R2adj))")
    println("Residuals. StDev: $(myround(reg.resStDev)), Min: $(myround(minimum(reg.res))), Max: $(myround(maximum(reg.res)))")
    println("")

    V = isnan(reg.intercept) ? reg.gmm.coefCov : reg.gmm.coefCov[2:end, 2:end]
    wald = reg.nobs * reg.coef' * inv(V) * reg.coef
    waldPVal = 1 .- cdf(Chisq(length(reg.coef)), wald)
    println("Wald on Joint Significance: $(myround(wald)), pval: $(myround(waldPVal))")
    println("Akaike IC: $(myround(reg.AIC)), Bayesian IC: $(myround(reg.BIC))")

    return
end

"""
        multiOLS(y, x; <kwargs>)

Report table with estimates of `y = x β + e` by OLS using multiple subsets of explanatory variable in `x`. See keyword `subsets`. 

### Arguments
- `y::Vector{Float}`: vector with explained variable sample
- `x::VecOrMat{Float}`: array with explanatory variable(s) sample (dim 1 = observations, dim 2 = variables)

### Keyword Arguments
- `intercept::Bool`: whether to add intercept term as regressor
- `subsets::Vector{Vector{Int}}`: indexes of explanatory variables in each regression specification. Default = [1:size(x, 2)] (only the subset of all `x`)
- `spectral_model`: estimator of spectral density of moment sample (see documentation for `gmm` function). Choose between: 
    - `preset(S)` for a given `S::Array{Float64, 3}` (dim 1: observations, dim 2: moments, dim 3: parameters)
    - `nw(k)` (Newey & West 1987), where `k::Int64` is the number of lags 
    - `hh(k)` (Hansen & Hodrick 1980), where `k::Int64` is the number of lags 
    - `white()` (White (1980), serially uncorrelated `f`) 
    Default = `white()` 
"""
function multiOLS(y::Array{Float64,1},
    x::J where {J<:VecOrMat{Float64}};
    intercept::Bool=true,
    subsets::Vector{Vector{Int64}}=[[1:size(x, 2)...]],
    spectral_model=white())

    coefTable = DataFrame(
        ["var$p" => String[] for p in axes(x, 2)]...,
        "R2adj" => Float64[],
        "AIC" => Float64[],
    )

    intercept && (coefTable = hcat(DataFrame("int" => String[]), coefTable))

    for subset in subsets
        reg = regOLS(y, x[:, subset]; intercept=intercept, spectral_model=spectral_model)

        row = []
        str(x) = string(round(x, digits=3))
        if intercept
            term = str(reg.intercept) * " " * stars(compute_pval(reg.intercept / sqrt(reg.gmm.coefCov[1, 1] / reg.nobs)))
            push!(row, term)
        end

        for globalIdx in axes(x, 2)
            if globalIdx in subset
                localIdx = findfirst(x -> x == globalIdx, subset)
                term = str(reg.coef[localIdx]) * " " * stars(compute_pval(reg.tStat[localIdx]))
            else
                term = " "
            end
            push!(row, term)
        end
        push!(row, reg.R2adj)
        push!(row, reg.AIC)
        push!(coefTable, row)
    end

    header = [["Var $p" for p in axes(x, 2)]...; "R2 Adj"; "AIC"]
    intercept && (header = ["(Intercept)"; header])

    println("")
    pretty_table(coefTable;
        title="OLS Results",
        tf=tf_compact,
        header=header,
        formatters=ft_printf("%7.3f"),
        alignment=:c
    )
    println("Significance: 1% (***) 2.5% (**) 5% (*) 10% (.)")
    println("")

    return
end

"""
        multiIV(y, x, z; <kwargs>)

Report table with estimates of `y = x β + e` through IV variables `z` using multiple subsets of explanatory variable in `x`. See keyword `subsets`. 

The order condition requires the number of instruments `z` to be as least as many as the number of explanatory variables `x`.

### Arguments
- `y::Vector{Float}`: vector with explained variable sample
- `x::VecOrMat{Float}`: array with explanatory variable(s) sample (dim 1 = observations, dim 2 = variables)
- `z::VecOrMat{Float}`: array with instrument(s) sample (dim 1 = observations, dim 2 = variables)

### Keyword Arguments
- `intercept::Bool`: `true` to add intercept term, both as regressor and as instrument
- `two_step::Bool`: estimate `β` two times, using efficient weights in the second run  
- `weight::Matrix`: weighting matrix for orthogonality conditions. Default = identity
- `subsets::Vector{Vector{Int}}`: indexes of explanatory variables in different specifications of the regression to be estimated. Default = [1:size(x, 2)] (only the subset of all `x`)
- `spectral_model`: estimator of spectral density of moment sample (see documentation for `gmm` function). Choose between: 
    - `preset(S)` for a given `S::Array{Float, 3}` (dim 1: observations, dim 2: moments, dim 3: parameters)
    - `nw(k)` (Newey & West 1987), where `k::Int64` is the number of lags 
    - `hh(k)` (Hansen & Hodrick 1980), where `k::Int64` is the number of lags 
    - `white()` (White (1980), serially uncorrelated `f`) 
    Default = `white()` 
"""
function multiIV(y::Array{Float64,1},
    x::J where {J<:VecOrMat{Float64}},
    z::H where {H<:VecOrMat{Float64}};
    intercept::Bool=true,
    two_step::Bool=false,
    weight::Matrix{Float64}=diagm(ones(size(z, 2) + intercept)),
    subsets::Vector{Vector{Int64}}=[[1:size(x, 2)...]],
    spectral_model=white())

    coefTable = DataFrame(
        ["var$p" => String[] for p in axes(x, 2)]...,
        "R2adj" => Float64[],
        "AIC" => Float64[],
        "J" => String[],
    )

    intercept && (coefTable = hcat(DataFrame("int" => String[]), coefTable))

    for subset in subsets
        reg = regIV(y, x[:, subset], z;
            intercept=intercept,
            two_step=two_step,
            weight=weight,
            spectral_model=spectral_model
        )

        row = []
        str(x) = string(round(x, digits=3))
        if intercept
            term = str(reg.intercept) * " " * stars(compute_pval(reg.intercept / sqrt(reg.gmm.coefCov[1, 1] / reg.gmm.nobs)))
            push!(row, term)
        end

        for globalIdx in axes(x, 2)
            if globalIdx in subset
                localIdx = findfirst(x -> x == globalIdx, subset)
                term = str(reg.coef[localIdx]) * " " * stars(compute_pval(reg.tStat[localIdx]))
            else
                term = " "
            end
            push!(row, term)
        end
        push!(row, reg.R2adj)
        push!(row, reg.AIC)
        if reg.gmm.nmom > reg.gmm.npar
            pval_J = 1 - cdf(Chisq(reg.gmm.nmom - reg.gmm.npar), reg.gmm.nobs * reg.gmm.J)
        else
            pval_J = 1.0
        end
        push!(row, str(reg.nobs * reg.gmm.J) * " " * stars(pval_J))

        push!(coefTable, row)
    end

    header = [["Var $p" for p in axes(x, 2)]...; "R2 Adj"; "AIC"; "T × J"]
    intercept && (header = ["(Intercept)"; header])

    println("")
    pretty_table(coefTable;
        title="IV Results",
        tf=tf_compact,
        header=header,
        formatters=ft_printf("%7.3f"),
        alignment=:c
    )
    println("Significance: 1% (***) 2.5% (**) 5% (*) 10% (.)")
    println("J tests overidentifying restrictions assuming efficient weights")
    println("Moment conditions: $(size(z, 2)+intercept)")
    println("")

    return
end
