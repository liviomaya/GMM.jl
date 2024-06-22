using Distributions, Plots, LinearAlgebra, Statistics
using GMM, Optim


# test regression methods (regOLS, regIV, multiOLS, multiIV, report):
function regression_test()
    #= 

    The true model is
        y = β X + ϵ
    where X ∼ [1 𝑁(0, Ω)] and std(ϵ) = σ

    We are given instruments Z

    The following moment conditions hold:
        E(X ⊗ ϵ) = E(X ⊗ (y - β X)) = 0 
        E(Z ⊗ ϵ) = E(Z ⊗ (y - β X)) = 0 

    I use these moment conditions to estimate β. 
    =#

    T = 200 # sample size
    β = [10.0, 0.8, -0.2] # true β
    σ = 0.80 # true σ

    # simulate X
    Ω = [1.0 0.20; 0.20 0.50]
    dist_X = MvNormal(zeros(2), Ω)
    X = [ones(T) rand(dist_X, T)']

    # simulate Y
    dist_e = Normal(0.0, σ)
    e = rand(dist_e, T)
    Y = X * β + e

    # simulate Z
    nu = rand(dist_e, T, 2)
    nu2 = rand(dist_e, T, 2)
    Z = [ones(T) (X[:, [2, 3]] .+ nu) (X[:, [2, 3]] .+ nu2)]

    #########################################################
    # (By Hand) OLS
    #########################################################
    b_OLS = inv(X' * X) * X' * Y
    eOLS = Y .- X * b_OLS
    ΣOLS = cov(eOLS, corrected=false)
    bCov_OLS = inv(X' * X / T) * ΣOLS

    #########################################################
    # (Package) Regression Functions
    #########################################################
    ols = regOLS(Y, X[:, [2, 3]]; intercept=true)
    ivE = regIV(Y, X[:, [2, 3]], Z[:, 2:3]; intercept=true)
    ivO = regIV(Y, X[:, [2, 3]], Z[:, 2:end]; intercept=true)

    #########################################################
    # (Package) GMM estimation functions
    #########################################################
    # Moment condition: E(X ⊗ ϵ) = E(X ⊗ (y - β X)) = 
    f(b) = X .* ((Y .- X * b) * ones(size(X, 2))')
    b0 = [5.0, 0.0, 0.0]
    gmmS = gmm(f, b0; df=forwarddiff(), opt=Optim.Options(show_trace=false))


    #########################################################
    # Reporting
    ############################################
    println(" ")
    println(" ")
    println("Coefficients estimates:")
    println("(starting from true β)")
    println(" ")
    display([β b_OLS ols.gmm.coef ivE.gmm.coef ivO.gmm.coef gmmS.coef])

    println(" ")
    println(" ")
    println(" ")

    println("Asymptotic variance:")
    println(" ")
    println("OLS (by hand):")
    display(bCov_OLS)
    println(" ")
    println("OLS (package):")
    display(ols.gmm.coefCov)
    println(" ")
    println("IV Overidentified (package):")
    display(ivO.gmm.coefCov)
    println(" ")
    println("GMM (package):")
    display(gmmS.coefCov)
    println(" ")
    println(" ")
    println(" ")

    #########################################################
    # Multi-Regression Tables
    ############################################

    report(ols)

    multiOLS(Y, X[:, [2, 3]];
        intercept=true,
        subsets=[[1, 2], [1], [2]],
        spectral_model=white())

    multiIV(Y, X[:, [2, 3]], Z[:, 2:end];
        intercept=true,
        two_step=false,
        weight=inv(ivO.gmm.spectral),
        subsets=[[1, 2], [1], [2]],
        spectral_model=preset(ivO.gmm.spectral))

    return nothing
end
@time regression_test()

# test multivariate regression
function mv_reg_test()

    function generate_some_data()

        T = 200 # sample size
        β1 = [10.0, 0.8, -0.2, 0.2] # true β
        β2 = [5.0, 0.1, 1.0, -0.50] # true β
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
    Y, x, B0 = generate_some_data()
    T, N = size(Y)
    X = [ones(T) x]
    Z = X
    P = size(x, 2) + 1

    weight = diagm(ones(2 * 4))
    iv = regIV(Y, x[:, 1], x;
        intercept=true,
        two_step=false,
        weight=weight,
        spectral_model=white())
    # display([vec(B0[:, 1:2]') vec([iv.intercept iv.coef]')])

    ols = regOLS(Y, x;
        intercept=true,
        spectral_model=white())
    # display([vec(B0') vec([ols.intercept ols.coef]')])

    report(ols)
    report(iv)
    println("")
    println("True Coefficients:")
    display(B0')
end
@time mv_reg_test()

# test gmm:
function estim_Gamma()
    # estimate Gamma parameter distribution matching mean, variance and skewness

    # estimate Gamma distribution with parameters
    T = 10000 # sample size
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
    coef0 = [0.2, 0.2]

    opt = Optim.Options(
        iterations=100,
        show_trace=true,
        show_every=10
    )

    sol = gmm(f, coef0;
        two_step=true,
        weight=diagm(ones(size(f(coef0), 2))),
        opt_steps=:default,
        df=forwarddiff(),
        spectral_model=white(),
        algorithm=BFGS(),
        opt=opt)

    #########################################################
    # Reports
    #########################################################
    println(" ")
    println(" ")
    println("Coefficients estimate:")
    println(" ")
    println("True: $([α, β])")
    println("Estm: $(round.(sol.coef, digits=2))")

    return sol
end
@time estim_Gamma();