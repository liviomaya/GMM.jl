# ]activate "Project.toml"
using Distributions, Plots, LinearAlgebra, Statistics
using GMM


# test gmm_spec and reg
function regression_test()
    #= 

    The true model is
        y = β X + ϵ
    where X ∼ [1 𝑁(0, Ω)].

    The following moment conditions hold:
        E(X ⊗ ϵ) = E(X ⊗ (y - β X)) = 0 
        E(ϵϵ') = Σ 

    In this example, I use the first moment condition above to estimate β. 
    =#

    T = 200 # sample size
    β = [10.0, 0.8, -0.2] # true β
    σ = 0.20 # true σ

    # simulate X
    Ω = [1.0 0.20; 0.20 0.50]
    dist_X = MvNormal(zeros(2), Ω)
    X = [ones(T) rand(dist_X, T)']
    # scatter(X[:, 2], X[:, 3], title = "X") |> display

    # simulate e
    dist_e = Normal(0.0, σ)
    e = rand(dist_e, T)
    # scatter(X[:, 2], e, title="e") |> display

    # left-hand variable
    Y = X * β + e
    # scatter(X[:, 2], Y[:, 1], title="Y") |> display

    #########################################################
    # Test reg_table
    #########################################################
    println("Regression Table:")
    reg_table(Y, X[:, 2:3]; subset=[[1, 2], [1], [2]], intercept=true, s=white(), R2adj=true)


    #########################################################
    # Calculate OLS by hand 
    #########################################################
    b_OLS = (Y' * X) * inv(X' * X) |> vec
    eOLS = Y .- X * b_OLS
    ΣOLS = cov(eOLS, corrected=false)
    bCov_OLS = inv(X' * X / T) * ΣOLS

    #########################################################
    # Use package functions
    #########################################################
    b_REG, bCov_REG, e, R2 = reg(Y, X[:, 2:3]; intercept=true, s=white(), R2adj=true)


    #########################################################
    # Use GMM estimation functions
    #########################################################
    # Moment condition: E(X ⊗ ϵ) = E(X ⊗ (y - β X)) = 
    f(b) = X .* ((Y .- X * b) * ones(size(X, 2))')

    b0 = [5.0, 0.0, 0.0]
    sol = gmm(f, b0; df=finite_diff(), s=hansen_hodrick(5))

    #########################################################
    # Reports
    #########################################################
    println(" ")
    println(" ")
    println("Coefficients estimate:")
    println(" ")
    println("True: $β")
    println("OLS (by hand): $b_OLS")
    println("OLS (package): $b_REG")
    println("GMM (package): $(sol.b)")

    println(" ")
    println(" ")

    println("Asymptotic variance:")
    println(" ")
    println("OLS (by hand):")
    display(bCov_OLS)
    println(" ")
    println("OLS (package - White):")
    display(bCov_REG)
    println(" ")
    println("GMM (package - Hansen & Hodrick):")
    display(sol.bCov)
    println(" ")

    return nothing
end
regression_test()

# test multivariate regression
function mv_reg_test()

    function generate_some_data()

        T = 200
        β1 = [10.0, 0.8, -0.2, 0.2]
        β2 = [5.0, 0.1, 1.0, -0.50]
        σ = 2.0 # true σ

        # simulate X
        Ω = [1.0 0.20 0.0; 0.20 0.50 0.0; 0.0 0.0 1.0]
        dist_X = MvNormal(zeros(3), Ω)
        x = permutedims(rand(dist_X, T))
        X = [ones(T) x]

        # simulate e
        dist_e = Normal(0.0, σ)
        e1 = rand(dist_e, T)
        e2 = rand(dist_e, T)

        y1 = X * β1 + e1
        y2 = X * β2 + e2
        y = [y1 y2]
        B0 = [β1'; β2']

        return y, x, B0
    end
    y, x, B0 = generate_some_data()
    T, N = size(y)
    P = size(x, 2) + 1

    B, bCov, e, R2 = mv_reg(y, x; intercept=true, s=white())

    println("True B:")
    display(round.(B0, digits=1))
    println("")
    println("Estimated B:")
    display(round.(B, digits=1))
    println("")
    println("Standard Deviation B:")
    stdB = sqrt.(reshape(diag(bCov), P, N)' / T)
    display(round.(stdB, digits=2))
    println("")
    println("R-Squared:")
    display(round.(R2, digits=2))

    return
end
mv_reg_test()

# test gmm:
function estim_Gamma()
    # estimate Gamma parameter distribution matching mean, variance and skewness

    # estimate Gamma distribution with parameters
    T = 1000 # sample size
    α = 1.0
    β = 2.0

    # simulate X
    dist_X = Gamma(α, 1 / β)
    X = rand(dist_X, T)

    # moment conditions
    function f(a, b)
        (a < 0) && return 1e8 * ones(size(X, 1), 3)
        (b < 0) && return 1e8 * ones(size(X, 1), 3)

        μ = a / b
        σ = sqrt(a / b^2)
        κ = 2 / sqrt(a) # skewness

        m_mean = X .- μ
        m_var = (X .^ 2) .- μ^2 .- σ^2
        m_skew = (X .^ 3) .- ((σ^3) * κ + 3 * μ * σ^2 + μ^3)

        m = [m_mean m_var m_skew]
        return m
    end
    f(p) = f(p[1], p[2])

    # optimization
    b0 = [0.2, 0.2]
    N = 2

    sol = gmm(f, b0;
        N=N,
        W=diagm(ones(size(f(b0), 2))),
        opt_steps=:default,
        df=finite_diff(),
        s=newey_west(10),
        algorithm=BFGS(),
        iterations=100,
        show_trace=true,
        show_every=10)

    #########################################################
    # Reports
    #########################################################
    println(" ")
    println(" ")
    println("Coefficients estimate:")
    println(" ")
    println("True: $([α, β])")
    println("Estm: $(round.(sol.b, digits=2))")

    return
end
estim_Gamma()