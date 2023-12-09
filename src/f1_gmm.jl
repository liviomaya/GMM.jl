"""
    GMMSolution(b, g, bCov, gCov, Dg, S, J)

Stores the solution to an GMM model (see `gmm` function).

### Arguments

- `b::Vector{Float64}`: GMM parameter estimate.

- `g::Vector{Float64}`: sample moments.

- `bCov::Matrix{Float64}`: asymptotic covariance (parameters).

- `gCov::Matrix{Float64}`: asymptotic covariance (moments).

- `Dg::Matrix{Float64}`: derivative of sample moments.

- `S::Matrix{Float64}`: asymptotic covariance (sample moments); spectral density.

- `J::Float64`: `𝐽` statistic `gᵀ W g` (Hansen (1982)'s `𝐽` statistics when `N` > 1)

### Asymptotic Distributions

√T (`b` - b₀) → 𝑁(0, `bCov`)

√T `g` → 𝑁(0, `gCov`) (`gCov` non-invertible, use `pinv` for inference)

T `J` → 𝜒²(M-P) (only when `W` → `S⁻¹` ; M moments, P parameters)
"""
struct GMMSolution
    b::Vector{Float64}
    g::Vector{Float64}
    bCov::Matrix{Float64}
    gCov::Matrix{Float64}
    Dg::Matrix{Float64}
    S::Matrix{Float64}
    J::Float64
end

g_apply(f, b) = mean(f(b), dims=1)[:]

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

# optimize by increasingly growing dimensionality of the opt problem 
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
end

# search for optimal parameter vector
function minimize_objective(f, W, b0, opt_steps, algorithm, opt)
    function obj(b)
        B = g_apply(f, b)
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
function complementary_objects(f, W, b, df, s)
    M = size(W, 1)
    P = length(b)

    # calculate g, dg/db and S = T Var(g)
    g = g_apply(f, b)
    DF = df(f)
    Dg = mean(DF(b), dims=1) |> x -> reshape(x, (M, P))
    S = s(f, b)

    # check sensitivity 
    flag = any(eachcol(Dg) .== Ref(zeros(M)))
    if flag
        printstyled("Warning! "; color=:red, bold=true)
        printstyled("Singularity in Dg. Moments/parameters possibly misspecified. \n")
    end
    H = pinv(Dg' * W * Dg)

    # asymptotic distributions
    bCov = H * Dg' * W * S * W' * Dg * H'
    gCov = (I(M) - Dg * H * Dg' * W) * S * (I(M) - Dg * H * Dg' * W)'

    # test over-identified restrictions
    J = g' * W * g
    return g, bCov, gCov, Dg, S, J
end



# √T (b̂ - b) → 𝑁(0, bCov)
# √T g → 𝑁(0, gCov) (gCov non-invertible, use pinv instead)
# T J_stat → Χ²(M-P)
function gmm_step(f::Function,
    b0::Vector{Float64},
    W::Matrix{Float64},
    df,
    s,
    opt_steps,
    algorithm,
    iterations,
    show_trace,
    show_every)

    # optimizer options
    opt = Optim.Options(
        iterations=iterations,
        show_trace=show_trace,
        show_every=show_every
    )

    # minimize gᵀ W g
    b = minimize_objective(f, W, b0, opt_steps, algorithm, opt)

    # compute other objects
    g, bCov, gCov, Dg, S, J = complementary_objects(f, W, b, df, s)

    return b, g, bCov, gCov, Dg, S, J
end

"""
    sol = gmm(f, b0; <kwargs>)

Solve the GMM model `Min E[f(b)]' W E[f(b)]`. Returns `GMMSolution` object.

### Arguments

- `f::Function`: moment function; `f(b)` returns observations in rows.

- `b0::Vector{Float64}`: initial guess for optimization algorithm.


### Keyword Arguments

- `N::Int64`: number of iterations to re-estimate optimal weighting matrices `W`. Default = 1.

- `W::Matrix{Float64}`: Weighting matrix. Defaults to identity.

- `df`: derivative of function `f`. Two options: `exact(df)` for a given function `df(x,b)` or `forwarddiff(;step=1e-5)` for forward automatic differentiation. Default = `forwarddiff()`.

- `s`: matrix `S` is an estimate of the asymptotic variance of the sample mean `∑ E[f(b) f(b)ᵀ]`. Algorithms for calculations: `preset(S)` for a given `S`, `nw(k)` (Newey & West 1987) or `hh(k)` (Hansen & Hodrick 1980) for given number of lags, or still `white()` (serially uncorrelated `f`). Default = `white()`.

- `opt_steps`: If `:default`, search directly over space of `b` using `b0` as initial guess. If `:growing`, search over space `b[1:p]`, growing `p` iteratively and using as starting condition the optimized value of the previous iteration. If `:univariate`, search over `b[p]` space fixing `b[1:p-1]` on optimized values; then search directly over space of `b` using as initial guess the resulting vector.

- `algorithm`: numerical optimization algorithm (see `Optim` package). Use `BFGS()` or `Newton()`. Default = `BFGS()`.

- `iterations::Int64`: Optimization parameter. Number of iterations. Default=100.

- `show_trace::Bool`: Optimization parameter. Determines whether to show trace. Default=`false`.

- `show_every::Int64`: Optimization parameter. Interval between trace reports. Default=10.

"""
function gmm(f::Function,
    b0::Vector{Float64};
    N::Int64=1,
    W::Matrix{Float64}=diagm(ones(size(f(b0), 2))),
    df=forwarddiff(),
    s=white(),
    opt_steps=:default,
    algorithm=BFGS(),
    iterations::Int64=100,
    show_trace::Bool=false,
    show_every::Int64=10)

    b, g, bCov, gCov, Dg, S, J = gmm_step(f, b0, W, df, s, opt_steps, algorithm, iterations, show_trace, show_every)

    for n in 2:N
        if n < N
            s0 = s # recalculate every step prior to last
        elseif n == N
            s0 = preset(S)
        end
        b, g, bCov, gCov, Dg, S, J = gmm_step(f, b, inv(S), df, s0, opt_steps, algorithm, iterations, show_trace, show_every)
    end

    sol = GMMSolution(b, g, bCov, gCov, Dg, S, J)
    return sol
end
