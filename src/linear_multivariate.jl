
"""
        MvRegression

## Fields
- `gmm::GMMSolution`: solution to GMM problem on orthogonality conditions
- `neq::Int64`: number of equation
- `nreg::Int64`: number of regressors (includes intercept)
- `nobs::Int64`: sample size
- `intercept::Vector{Float64}`: equation intercept (`NaN` if no intercept)
- `coef::Matrix{Float64}`: estimated coefficients (format `neq`×`nreg` plus intercept)
- `stDev::Matrix{Float64}`: asymptotic standard deviation of `coef`
- `tStat::Matrix{Float64}`: `coef` / `stDev`
- `resCov::Matrix{Float64}`: MLE estimate of residuals' covariance
- `fitted::Matrix{Float64}`: fitted values
- `res::Matrix{Float64}`: sample residuals
- `SSE::Vector{Float64}`: sum of squared residuals
- `R2::Vector{Float64}`: R-Squared
- `R2adj::Vector{Float64}`: adjusted R-Squared
- `logLLH::Float64`: log-likelihood (assumes no residual serial correlation)
- `AIC::Float64`: Akaike information criterion (assumes no residual serial correlation)
- `BIC::Float64`: Bayesian information criterion (assumes no residual serial correlation)
"""
struct MvRegression
    gmm::GMMSolution
    neq::Int64
    nreg::Int64
    nobs::Int64
    intercept::Vector{Float64}
    coef::Matrix{Float64}
    stDev::Matrix{Float64}
    tStat::Matrix{Float64}
    resCov::Matrix{Float64}
    fitted::Matrix{Float64}
    res::Matrix{Float64}
    SSE::Vector{Float64}
    R2::Vector{Float64}
    R2adj::Vector{Float64}
    logLLH::Float64
    AIC::Float64
    BIC::Float64
end

# column j of B with coefficients from equation j
function mv_orthogonal(Y, X, Z, B::Matrix{Float64})
    e = Y .- X * B
    mom = zeros(size(Y, 1), size(Z, 2) * size(Y, 2))
    for i in axes(Y, 1)
        j = 1
        for colY in axes(Y, 2)
            for colZ in axes(Z, 2)
                mom[i, j] = Z[i, colZ] * e[i, colY]
                j += 1
            end
        end
    end
    return mom
end

function mv_orthogonal(Y, X, Z, b::Vector{Float64})
    B = reshape(b, size(X, 2), size(Y, 2))
    return mv_orthogonal(Y, X, Z, B)
end

function gaussian_loglikelihood(X::Matrix{Float64}, Mu::Vector{Float64}, Sigma::Matrix{Float64})
    T, N = size(X)
    res = [X[i, j] - Mu[j] for i in axes(X, 1), j in axes(X, 2)]
    logL = -(T * N / 2) * log(2 * pi)
    logL += -(T / 2) * log(det(Sigma))
    iSigma = inv(Sigma)
    for i in 1:T
        logL += -res[i, :]' * iSigma * res[i, :] / 2
    end
    return logL
end

function mv_gmmiv_exact(Y, X, Z, spectral_model)
    T, nZ = size(Z)
    nY = size(Y, 2)
    coef = (Z' * X) \ Z' * Y
    mom = zeros(nZ * nY)
    Dmom = -kron(diagm(ones(nY)), (Z' * X) / T) # - E(zₜxₜᵀ)
    momCov = zeros(nZ * nY, nZ * nY)
    f(b) = mv_orthogonal(Y, X, Z, b)
    spectral = spectral_model(f, coef[:])
    coefCov = inv(Dmom) * spectral * inv(Dmom)'
    weight = diagm(ones(nZ * nY))
    return GMMSolution(coef[:], mom, coefCov, momCov, Dmom, spectral, weight, T)
end

function mv_gmmiv_over(Y, X, Z, weight, spectral_model)
    T, nZ = size(Z)
    nY = size(Y, 2)
    sZY = reduce(vcat, [Z' * y for y in eachcol(Y)]) ./ T
    sZX = kron(diagm(ones(nY)), Z' * X) ./ T
    coef = (sZX' * weight * sZX) \ sZX' * weight * sZY
    mom = sZY .- sZX * coef
    Dmom = -sZX # - E(zₜxₜᵀ)
    f(b) = mv_orthogonal(Y, X, Z, b)
    spectral = spectral_model(f, coef)
    H = inv(Dmom' * weight * Dmom)
    coefCov = H * Dmom' * weight * spectral * weight' * Dmom * H'
    momCov = (I(nZ * nY) - Dmom * H * Dmom' * weight) * spectral * (I(nZ * nY) - Dmom * H * Dmom' * weight)'
    return GMMSolution(coef, mom, coefCov, momCov, Dmom, spectral, weight, T)
end

function build_mv_regression(gmmSol, Y, X, intercept)
    nY, nX = size(Y, 2), size(X, 2)
    coef = permutedims(reshape(gmmSol.coef, nX, nY))
    stDev = permutedims(reshape(sqrt.(diag(gmmSol.coefCov) / gmmSol.nobs), nX, nY))
    tStat = coef ./ stDev
    fitted = X * coef'
    res = Y .- fitted
    resCov = (res' * res) / gmmSol.nobs
    SSE = [sum(r .^ 2) for r in eachcol(res)]
    R2 = [1 - SSE[j] / sum((y .- mean(y)) .^ 2) for (j, y) in enumerate(eachcol(Y))]
    R2adj = 1 .- (1 .- R2) * (gmmSol.nobs - 1) ./ (gmmSol.nobs - nX)
    logLLH = gaussian_loglikelihood(res, zeros(nY), resCov)

    # for AIC, check https://robjhyndman.com/hyndsight/lm_aic.html
    npar = gmmSol.npar + nY^2 # add parameters in Cov(res)
    AIC = 2 * npar - 2 * logLLH
    BIC = log(gmmSol.nobs) * npar - 2 * logLLH

    return MvRegression(gmmSol,
        nY, nX, gmmSol.nobs,
        intercept ? coef[:, 1] : NaN * ones(nY),
        intercept ? coef[:, 2:end] : coef,
        intercept ? stDev[:, 2:end] : stDev,
        intercept ? tStat[:, 2:end] : tStat,
        resCov,
        fitted,
        res,
        SSE,
        R2,
        R2adj,
        logLLH,
        AIC,
        BIC)
end

function regOLS(Y::Matrix{Float64},
    x::J where {J<:VecOrMat{Float64}};
    intercept::Bool=true,
    spectral_model=white())

    return regIV(Y, x, x;
        intercept=intercept,
        spectral_model=spectral_model)
end

function regIV(Y::Matrix{Float64},
    x::J where {J<:VecOrMat{Float64}},
    z::H where {H<:VecOrMat{Float64}};
    intercept::Bool=true,
    two_step::Bool=false,
    weight::Matrix{Float64}=diagm(ones(size(z, 2) + intercept)),
    spectral_model=white())

    X = intercept ? [ones(size(x, 1)) x] : x
    Z = intercept ? [ones(size(x, 1)) z] : z

    if size(z, 2) == size(x, 2)
        gmmSol = mv_gmmiv_exact(Y, X, Z, spectral_model)
    elseif size(z, 2) > size(x, 2)
        use_weight = weight
        gmmSol = mv_gmmiv_over(Y, X, Z, use_weight, spectral_model)

        if two_step
            use_weight = inv(gmmSol.spectral)
            gmmSol = mv_gmmiv_over(Y, X, Z, use_weight, preset(gmmSol.spectral))
        end
    else
        error("Order condition violated. Additional instruments required.")
    end

    return build_mv_regression(gmmSol, Y, X, intercept)
end

function report(reg::MvRegression)

    intercept_var = reshape(diag(reg.gmm.coefCov), reg.nreg, reg.neq)[1, :]

    myround(x) = round(x, digits=3)

    for eq in 1:reg.neq
        coefTable = DataFrame(
            "variable" => String[],
            "coef" => Float64[],
            "std" => Float64[],
        )

        # table
        if !isnan(reg.intercept[1]) # check for intercept
            push!(coefTable, ["(Intercept)" reg.intercept[eq] sqrt(intercept_var[eq] / reg.nobs)])
        end

        for p in axes(reg.coef, 2)
            push!(coefTable, ["Variable $p" reg.coef[eq, p] reg.stDev[eq, p]])
        end
        coefTable.tstat = coefTable.coef ./ coefTable.std
        coefTable.pval = compute_pval.(coefTable.tstat)
        coefTable.stars = stars.(coefTable.pval)

        println("")
        pretty_table(coefTable;
            title="Equation $eq, Coefficients",
            tf=tf_compact,
            header=["", "Estim", "StDev", "tStat", "P(>|t|)", ""],
            formatters=ft_printf("%7.3f", 2:5)
        )
        println("Significance: 1% (***) 2.5% (**) 5% (*) 10% (.)")
        println("")
        println("R-Squared: $(myround(reg.R2[eq])), Adjusted R-Squared: $(myround(reg.R2adj[eq]))")
        println("Residuals. StDev: $(myround(sqrt(reg.resCov[eq, eq]))), Min: $(myround(minimum(reg.res[:, eq]))), Max: $(myround(maximum(reg.res[:, eq])))")

        # Wald test of joint significance
        # restriction matrix
        indicator = diagm(ones(reg.neq))[:, eq]
        if !isnan(reg.intercept[1]) # check for intercept
            eye = diagm(ones(reg.nreg - 1))
            eye = [zeros(reg.nreg - 1) eye]
        else
            eye = diagm(ones(reg.nreg))
        end
        R = kron(indicator', eye)
        V = R * reg.gmm.coefCov * R'
        wald = reg.nobs * (R * reg.gmm.coef)' * inv(V) * (R * reg.gmm.coef)
        waldPVal = 1 .- cdf(Chisq(size(R, 1)), wald)

        println("Wald on Joint Significance: $(myround(wald)), pval: $(myround(waldPVal))")
        println("")
    end
    println("")
    println("Akaike IC: $(myround(reg.AIC)), Bayesian IC: $(myround(reg.BIC))")

    return
end
