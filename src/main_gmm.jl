"""
    GMMSolution(coef, mom, coefCov, momCov, Dmom, spectral, J, npar, nmom)

Stores the solution to an GMM model (see `gmm` function).

### Arguments

- `coef::Vector{Float64}`: GMM estimate of parameter vector.

- `mom::Vector{Float64}`: sample moments evaluated at `coef`.

- `coefCov::Matrix{Float64}`: asymptotic covariance (parameters).

- `momCov::Matrix{Float64}`: asymptotic covariance (moments).

- `Dmom::Matrix{Float64}`: sample moments differential.

- `spectral::Matrix{Float64}`: asymptotic covariance (sample moments); spectral density.

- `weight::Matrix{Float64}`: weighting matrix of the moment minimization problem.

- `J::Float64`: `𝐽` statistic `momᵀ weight mom` (Hansen (1982)'s `𝐽` statistic when weight -> inv(spectral))

- `npar::Int64`: number of parameters.

- `nmom::Int64`: number of moments.

- `nobs::Int64`: sample size.

### Asymptotic Distributions

√nobs × (`coef` - coef₀) → 𝑁(0, `coefCov`)

√nobs × `mom` → 𝑁(0, `momCov`) (`momCov` non-invertible, use `pinv` for inference)

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


# optimize by increasingly growing dimensionality of the opt problem
function opt_growing_dim(obj, b0, algorithm, opt)

    P = length(b0)
    x = zeros(0)
    x0 = zeros(0)
    for p in 1:P
        opt.show_trace && printstyled("Optimization - Dim $p of $P \n"; bold=true)
        temp_obj(X) = obj([X; b0[p+1:P]])
        x0 = vcat(x, b0[p])
        r = optimize(temp_obj, x0, algorithm, opt)
        x = r.minimizer

        if (p == P) && !Optim.converged(r)
            println("Minimization of gᵀ W g failed")
        end
    end
    return x
end

#= optimize by increasingly growing dimensionality of the opt problem 
function opt_multi_univariate(obj, b0, algorithm, opt)

    P = length(b0)
    x = zeros(0)
    x0 = zeros(0)
    for p in 1:P
        opt.show_trace && printstyled("Optimization - Parameter $p of $P \n"; bold=true)
        temp_obj(X) = obj([x; X; b0[p+1:P]])
        x0 = [b0[p]]
        r = optimize(temp_obj, x0, algorithm, opt)
        incr::Vector{Float64} = r.minimizer
        x = vcat(x, incr)
    end

    # solve complete problem
    opt.show_trace && printstyled("Optimization - Complete Problem \n"; bold=true)
    x0 = x
    r = optimize(obj, x0, algorithm, opt)
    !Optim.converged(r) && println("Minimization of gᵀ W g failed")
    x::Vector{Float64} = r.minimizer
    return x
end =#

# search for optimal parameter vector
function minimize_objective(f, W, b0, opt_steps, algorithm, opt)
    function obj(b)
        B = mean(f(b), dims=1)[:]
        return B' * W * B
    end
    if opt_steps == :default
        r = optimize(obj, b0, algorithm, opt)
        !Optim.converged(r) && println("Minimization of gᵀ W g failed")
        b = r.minimizer
    elseif opt_steps == :growing
        b = opt_growing_dim(obj, b0, algorithm, opt)
    elseif opt_steps == :univariate
        b = opt_multi_univariate(obj, b0, algorithm, opt)
    end
    return b
end

# compute remaining objects
function gmm_objects(f, weight, coef, df, spectral_model)
    nmom = size(weight, 1)
    npar = length(coef)

    # compute moments and spectral density
    mom = mean(f(coef), dims=1)[:]
    DF = df(f)
    Dmom = mean(DF(coef), dims=1) |> x -> reshape(x, (nmom, npar))
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
    opt_steps,
    algorithm,
    opt)

    # minimize gᵀ W g
    coef = minimize_objective(f, weight, coef0, opt_steps, algorithm, opt)

    # compute other objects
    mom, coefCov, momCov, Dmom, spectral = gmm_objects(f, weight, coef, df, spectral_model)

    return coef, mom, coefCov, momCov, Dmom, spectral
end

"""
    sol = gmm(f, coef0; <kwargs>)

Solve the GMM model `Min E[f(b)]' W E[f(b)]`. Returns `GMMSolution` object.

### Arguments

- `f::Function`: moment function; `f(b)` returns observations in rows.

- `coef0::Vector{Float64}`: initial guess for optimization algorithm.


### Keyword Arguments

- `N::Int64`: number of iterations to re-estimate optimal weighting matrices `W`. Default = 1.

- `weight::Matrix{Float64}`: Weighting matrix. Defaults to identity.

- `df`: derivative of function `f`. Two options: `exact(df)` for a given function `df(x,b)` or `forwarddiff(;step=1e-5)` for forward automatic differentiation. Default = `forwarddiff()`.

- `spectral_model`: estimator of `∑ E[f(b) f(b)ᵀ]`. Algorithms for calculations: `preset(S)` for a given `S`, `nw(k)` (Newey & West 1987) or `hh(k)` (Hansen & Hodrick 1980) for given number of lags, or still `white()` (serially uncorrelated `f`). Default = `white()`.

- `opt_steps`: If `:default`, search directly over space of `b` using `b0` as initial guess. If `:growing`, search over space `b[1:p]`, growing `p` iteratively and using as starting condition the optimized value of the previous iteration. If `:univariate`, search over `b[p]` space fixing `b[1:p-1]` on optimized values; then search directly over space of `b` using as initial guess the resulting vector.

- `algorithm`: numerical optimization algorithm (see `Optim` package). Use `BFGS()` or `Newton()`. Default = `BFGS()`.

- `iterations::Int64`: Optimization parameter. Number of iterations. Default=100.

- `show_trace::Bool`: Optimization parameter. Determines whether to show trace. Default=`false`.

- `show_every::Int64`: Optimization parameter. Interval between trace reports. Default=10.

"""
function gmm(f::Function,
    coef0::Vector{Float64};
    two_step::Bool=false,
    weight::Matrix{Float64}=diagm(ones(size(f(coef0), 2))),
    df=forwarddiff(),
    spectral_model=white(),
    opt_steps=:default,
    algorithm=BFGS(),
    opt=Optim.Options(
        iterations=1000,
        show_trace=true,
        show_every=25))

    use_weight = weight
    coef, mom, coefCov, momCov, Dmom, spectral = gmm_step(f, coef0, use_weight, df, spectral_model, opt_steps, algorithm, opt)

    if two_step
        use_weight = inv(spectral)
        coef, mom, coefCov, momCov, Dmom, spectral = gmm_step(f, coef, use_weight, df, preset(spectral), opt_steps, algorithm, opt)
    end

    return GMMSolution(coef, mom, coefCov, momCov, Dmom, spectral, use_weight, size(f(coef), 1))
end

