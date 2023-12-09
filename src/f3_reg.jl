
"""
`b, bCov, e, R2 = reg(y::Array{Float64,1}, x; constant=true, S=white(), R2adj=false)`

`s` can be `preset(x)` for a given `x`, `nw(k)` or `hh(k)` for given number of lags, or still `white()`

# Asymptotic Distribution

√T (b - b₀) → 𝑁(0, bCov)

"""
function reg(y::Array{Float64,1}, x::Matrix{Float64}; intercept=true, s=white(), R2adj=false)

    T = size(x, 1)

    # solve E(ϵxᵀ) = 0
    X = intercept ? [ones(T) x] : x
    b = inv(X' * X) * X' * y
    e = y .- X * b

    # calculate g, dg/db and S = T Var(g)
    dg = -(X' * X) / T # - E(xₜxₜᵀ)
    function f(β)
        T, P = size(X)

        mm = zeros(T, P)
        for t = 1:T
            ee = y[t] .- dot(β, X[t, :])
            mm[t, :] .= X[t, :] * ee
        end
        return mm
    end
    S = s(f, b)

    # asymptotic distributions
    bCov = inv(dg) * S * inv(dg)'
    bCov = Hermitian(bCov)
    # R2 = [e, demean(y)] .|> (z -> z.^2) .|> (z -> sum(z, dims=1)) |> (z -> 1 - z[1] / z[2])
    R2 = 1.0 - sum(e .^ 2) / sum(demean(y) .^ 2)
    R2adj && (R2 = 1 - (1 - R2) * (T - 1) / (T - size(x, 2) - 1))

    return b, bCov, e, R2
end

reg(y::Array{Float64,1}, x::Vector{Float64}; intercept=true, s=white(), R2adj=false) = reg(y, reshape(x, length(x), 1); intercept=intercept, s=s, R2adj=R2adj)

function reg_table(y, x;
    intercept=true,
    subset::Vector{Vector{Int64}}=[[1:size(x, 2)...]],
    s=white(),
    R2adj=false)
    @assert size(y, 2) == 1 "Method available only to single equation model"

    P = size(x, 2)
    C = length(subset)
    T = length(y)
    # data = NaN * ones(2 * C, P + 2)
    data::Array{Any} = fill("", 2 * C, P + 2 + intercept)
    for (c, comb) in enumerate(subset)
        b, bCov, e, R2 = reg(y, x[:, comb], intercept=intercept, s=s, R2adj=R2adj)
        σb = sqrt.(diag(bCov) / T)
        tb = (b ./ σb)
        p_val = 2 * (1 .- cdf(Normal(0, 1), abs.(tb)))

        I = (1+intercept):length(b)
        X2 = T * b[I]' * inv(bCov[I, I]) * b[I] # asymptotically equivalent to J test below 

        #= J test (assumes intercept):
            g = ([ones(T) x[:,comb]]' * demean(y) / T) 
            f(z, b) = z[end] * z[1:end - 1]
            S = s(f, b)
            J = T * g' * inv(s) * g =#
        Pcomb = length(comb)
        p_val_X2 = 1 .- cdf(Chisq(Pcomb), X2)

        i = 2 * c - 1
        rows = [i; i + 1]
        cols = intercept ? [1; comb .+ 1] : comb
        data[rows, cols] = [b'; p_val']
        data[rows, end-1] = [X2; p_val_X2]
        data[rows[1], end] = R2
    end

    header = [["β$(i - intercept)" for i = 1:(P+intercept)]; "𝜒²(𝛽=0)"; "R-Sq"]
    intercept && (header[1] = "α")
    formatters = ((v, i, j) -> ((j == P + 2 + intercept) && isodd(i)) ? 100 * v : v, ft_printf("%3.2f", 1:P+1+intercept), ft_printf("%3.1f", 1:P+2+intercept))
    h_coef = Highlighter((y, i, j) -> isodd(i), foreground=:blue, bold=true)
    h_tstat = Highlighter((y, i, j) -> iseven(i), foreground=:yellow)
    kw = [:header => header, :formatters => formatters, :vlines => [P + 1], :crop => :all, :highlighters => (h_coef, h_tstat)]
    println("")
    pretty_table(stdout, data; kw...)
    printstyled("p-values. "; color=:yellow)
    println("𝜒² tests null β=0.")
    println("")


    return nothing
end

"""
`B, bCov, e, R2 = mv_reg(y::Matrix{Float64}, x; intercept=true, S=white())`

`B[i,:]` stores the coefficients of the `i`-th equation.

`bCov`: asymptotic covariance matrix of `vec(B')`. Use `reshape(diag(bCov), P, N)'` for individual asymptotic variances, where `N` is the number of equations, `P` is the number of RHS variables.

`s` can be `preset(x)` for a given `x`, `nw(k)` or `hh(k)` for given number of lags, or still `white()`.

# Asymptotic Distribution

√T (vec(B') - b₀) → 𝑁(0, bCov)
"""
function mv_reg(y::Matrix{Float64}, x::Matrix{Float64}; intercept=true, s=white())

    T, N = size(y)
    @assert size(x, 1) == T
    X = intercept ? [ones(T) x] : x
    P = size(X, 2)
    M = N * P # number of moments = number of parameters

    B = y' * X * inv(X' * X)

    # moment: Xₜ ⊗ eₜ = Xₜ ⊗ (yₜ - (I ⊗ Xₜ') * b)
    # the first P entries of b refer to the first equation for y[:,1]
    function f(b)
        mom = zeros(T, M)
        for t in 1:T
            mom[t, :] = kron(X[t, :], y[t, :] .- kron(I(N), X[t, :]') * b)
        end
        #= same as
        B = reshape(b, P, N) |> permutedims
        e = y .- X * B'
        mom = reduce(hcat, [e[:,n].*X[:,p] for n in 1:N, p in 1:P])
        = =#
        return mom
    end

    # compute differential of moment
    function DF(b)
        dff = zeros(T, M, M)
        for t in 1:T
            dff[t, :, :] = kron(X[t, :], kron(I(N), X[t, :]'))
        end
        return dff
    end

    W = diagm(ones(M))
    df = exact(DF)
    b = B'[:]
    g, bCov = complementary_objects(f, W, b, df, s)[[1, 2]]
    e = y .- X * B'
    R2 = 1.0 .- sum(e .^ 2, dims=1)[:] ./ sum(demean(y) .^ 2, dims=1)[:]

    return B, bCov, e, R2
end

mv_reg(y::Matrix{Float64}, x::Vector{Float64}; intercept=true, s=white()) = mv_reg(y, reshape(x, length(x), 1); intercept=intercept, s=s)

"""
`pc, λ, V = principal_components(X; remove_mean=false, positive_eigvec=true)`

Option `positive_eigvec` forces most elements of eigenvectors to be positive

`pc[:,i]` stores the `i`-th principal component series

`λ` stores the eigenvalues in increasing order

`V[:,i]` contains `i`-th eigenvector
"""
function principal_components(X; remove_mean=false, positive_eigvec=true)
    remove_mean && (X = demean(X))
    λ, V = eigen(cov(X), sortby=z -> -z)

    if positive_eigvec
        Id = [count(col .< 0) > count(col .>= 0) for col in eachcol(V)]
        V .*= ones(size(V, 1)) * ((.!Id') .+ -1 * (Id'))
    end
    pc = X * V

    return pc, λ, V
end