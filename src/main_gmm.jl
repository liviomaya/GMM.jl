"""
    GMMSolution(coef, mom, coefCov, momCov, Dmom, spectral, J, npar, nmom)

Stores the solution to an GMM model (see `gmm` function).

### Fields

- `coef::Vector{Float}`: GMM estimate of parameter vector.

- `mom::Vector{Float}`: sample moments evaluated at `coef`.

- `coefCov::Matrix{Float}`: asymptotic covariance (parameters).

- `momCov::Matrix{Float}`: asymptotic covariance (moments).

- `Dmom::Matrix{Float}`: first derivative of sample moments.

- `spectral::Matrix{Float}`: asymptotic covariance (sample moments); spectral density.

- `weight::Matrix{Float}`: weighting matrix of the moment minimization problem.

- `J::Float`: `𝐽` statistic, or `momᵀ weight mom` (Hansen (1982)'s `𝐽` statistic when weight -> inv(spectral))

- `npar::Int`: number of parameters.

- `nmom::Int`: number of moments.

- `nobs::Int`: number of observations, or the sample size.

### Asymptotic Distributions

√n obs × (`coef` - coef₀) → 𝑁(0, `coefCov`)

√n obs × `mom` → 𝑁(0, `momCov`) (`momCov` non-invertible, use `pinv` for inference)

nobs × `J` → 𝜒²(nmom-npar) (if `weight` → `inv(spectral)` )
"""
struct GMMSolution
    coef::Vector{Float64}
    mom::Vector{Float64}
    coefCov::Matrix{Float64}
    momCov::Matrix{Float64}
    Dmom::Matrix{Float64}
    spectral::Matrix{Float64}
    weight::Matrix{Float64}
    J::Float64
    npar::Int64
    nmom::Int64
    nobs::Int64
end

GMMSolution(coef::Vector{Float64},
    mom::Vector{Float64},
    coefCov::Matrix{Float64},
    momCov::Matrix{Float64},
    Dmom::Matrix{Float64},
    spectral::Matrix{Float64},
    weight::Matrix{Float64},
    nobs::Int64
) = GMMSolution(coef, mom, coefCov, momCov, Dmom, spectral, weight, mom' * weight * mom, length(coef), length(mom), nobs)

function promote_to_matrix(vec_or_mat::VecOrMat)
    return vec_or_mat isa AbstractVector ? reshape(vec_or_mat, :, 1) : vec_or_mat
end

# search for optimal parameter vector
function minimize_objective(f, W, b0, algorithm, opt)
    function obj(b)
        B = mean(f(b), dims=1)[:]
        return B' * W * B
    end
    r = optimize(obj, b0, algorithm, opt)
    !Optim.converged(r) && println("Minimization of GMM objective failed.")
    return r.minimizer
end

# compute remaining objects
function gmm_objects(f, weight, coef, df, spectral_model)
    nmom = size(weight, 1)
    npar = length(coef)

    # compute moments and spectral density
    mom = mean(f(coef), dims=1)[:]
    DF = df(f)
    Dmom = reshape(mean(DF(coef), dims=1), (nmom, npar))
    spectral = spectral_model(f, coef)

    # check for singularity
    if any(eachcol(Dmom) .== Ref(zeros(nmom)))
        printstyled("Warning! "; color=:red, bold=true)
        printstyled("Singularity in `Dmom`. Moments/parameters possibly misspecified. \n")
    end

    # asymptotic distributions
    H = pinv(Dmom' * weight * Dmom)
    coefCov = H * Dmom' * weight * spectral * weight' * Dmom * H'
    momCov = (I(nmom) - Dmom * H * Dmom' * weight) * spectral * (I(nmom) - Dmom * H * Dmom' * weight)'

    # test over-identified restrictions
    # J = mom' * W * mom

    return mom, coefCov, momCov, Dmom, spectral
end



# √T (b̂ - b) → 𝑁(0, bCov)
# √T g → 𝑁(0, gCov) (gCov non-invertible, use pinv instead)
# T J_stat → Χ²(M-P)
function gmm_step(f::Function,
    coef0::Vector{Float64},
    weight::Matrix{Float64},
    df,
    spectral_model,
    algorithm,
    opt)

    # minimize gᵀ W g
    coef = minimize_objective(f, weight, coef0, algorithm, opt)

    # compute other objects
    mom, coefCov, momCov, Dmom, spectral = gmm_objects(f, weight, coef, df, spectral_model)

    return coef, mom, coefCov, momCov, Dmom, spectral
end

function check_consistency(y, weight)

    if length(y) == 0
        throw(ArgumentError("Moment function returns empty array."))
    end

    if size(weight, 1) != size(weight, 2)
        throw(ArgumentError("Weighting matrix must be square."))
    end

    if size(weight, 2) != size(y, 2)
        throw(ArgumentError("Size of weighting matrix not consistent with number of moments."))
    end

end

"""
    sol = gmm(f, coef0; <kwargs>)

Solve the generalized method of moments (GMM) problem `Min E[f(b)]' W E[f(b)]`. The output `sol` is a `GMMSolution` object.

### Arguments

- `f::Function`: moment function; `f(b)` returns observations in rows.

- `coef0::Vector{Float}`: initial guess for optimization algorithm.


### Keyword Arguments

- `two_step::Bool`: `true` to run optimization twice, using spectral matrix estimated in the first step to compute optimal weighting matrix. Default = false

- `weight::Matrix{Float}`: weighting matrix. Default = identity

- `df`: first derivative of moment function `f`. Choose between: 
    - `exact(df)` for a given function `df(x,b)`
    - `forwarddiff(; step=1e-5)` for a numerical forward differentiation algorithm
    Default = `forwarddiff()`

- `spectral_model`: estimator of `∑ E[f(b) f(b)ᵀ]`. Choose between: 
    - `preset(S)` for a given `S::Array{Float64, 3}` (dim 1: observations, dim 2: moments, dim 3: parameters)
    - `nw(k)` (Newey & West 1987), where `k::Int64` is the number of lags 
    - `hh(k)` (Hansen & Hodrick 1980), where `k::Int64` is the number of lags 
    - `white()` (White (1980), serially uncorrelated `f`) 
    Default = `white()`

- `algorithm`: optimization algorithm (see `Optim` package). Default = `BFGS()`

- `opt::Optim.Options`: options for optimization problem. See `Optim` package

"""
function gmm(f::Function,
    coef0::Vector{Float64};
    two_step::Bool=false,
    weight::Matrix{Float64}=diagm(ones(size(f(coef0), 2))),
    df=forwarddiff(),
    spectral_model=white(),
    algorithm=BFGS(),
    opt=Optim.Options(
        iterations=1000,
        show_trace=true,
        show_every=25))

    check_consistency(f(coef0), weight)

    # first step
    use_weight = weight
    coef, mom, coefCov, momCov, Dmom, spectral = gmm_step(f, coef0, use_weight, df, spectral_model, algorithm, opt)

    # second step
    if two_step
        use_weight = inv(spectral)
        coef, mom, coefCov, momCov, Dmom, spectral = gmm_step(f, coef, use_weight, df, preset(spectral), algorithm, opt)
    end

    return GMMSolution(coef, mom, coefCov, momCov, Dmom, spectral, use_weight, size(f(coef), 1))
end

