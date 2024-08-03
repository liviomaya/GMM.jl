@testset "General GMM" begin

    @testset "Argument Consistency" begin
        f(coef) = zeros(100, 2)
        weight = diagm(ones(3))
        coef0 = [1.0]

        @test_throws ArgumentError gmm(f, coef0; weight=zeros(2, 3))
        @test_throws ArgumentError gmm(f, coef0; weight=weight)
    end

    @testset "Solution Precision" begin
        # generate random Gamma distribution data
        n = 100000
        x = rand(Gamma(1.0, 1 / 5.0), n) # shape = 1.0, rate = 5.0

        # moments 
        function f(coef)
            a, b = coef
            (a < 0) && return 1e8 * ones(size(x, 1), 3)
            (b < 0) && return 1e8 * ones(size(x, 1), 3)

            μ = a / b
            σ = sqrt(a / b^2)
            κ = 2 / sqrt(a)

            M1 = x .- μ
            M2 = M1 .^ 2 .- σ^2
            M3 = (M1 ./ σ) .^ 3 .- κ
            return [M1 M2 M3]
        end

        # Run GMM optimization in a single step
        coef0 = [0.2, 0.2]
        m1 = gmm(f, coef0;
            two_step=false,
            df=forwarddiff(),
            spectral_model=white(),
            algorithm=BFGS(),
            opt=Optim.Options(show_trace=false)
        )

        # Run GMM optimization in two steps
        m2 = gmm(f, coef0;
            two_step=true,
            df=forwarddiff(),
            spectral_model=nw(4),
            algorithm=BFGS(),
            opt=Optim.Options(show_trace=false)
        )

        # Check sizes of solution object
        @test m1.npar == 2
        @test m1.nmom == 3
        @test m1.nobs == n
        @test length(m1.mom) == 3
        @test size(m1.momCov, 1) == size(m1.momCov, 2)
        @test size(m1.momCov, 1) == 3
        @test size(m1.Dmom, 1) == 3
        @test size(m1.Dmom, 2) == 2
        @test size(m1.spectral, 1) == size(m1.spectral, 2)
        @test size(m1.spectral, 1) == 3
        @test size(m1.weight, 1) == size(m1.weight, 2)
        @test size(m1.weight, 1) == 3
        @test size(m2.spectral, 1) == size(m2.spectral, 2)
        @test size(m2.spectral, 1) == 3
        @test size(m2.weight, 1) == size(m2.weight, 2)
        @test size(m2.weight, 1) == 3

        # Check that the estimated coefficients are close to the true values
        @test abs(m1.coef[1] - 1.0) < 0.03
        @test abs(m2.coef[1] - 1.0) < 0.03
        @test abs(m1.coef[2] - 5.0) < 0.1
        @test abs(m2.coef[2] - 5.0) < 0.1

        #  Check that the estimated coefficients are statistically significant
        @test all(abs.(m1.coef) ./ sqrt.(diag(m1.coefCov) / m1.nobs) .> 2)
        @test all(abs.(m2.coef) ./ sqrt.(diag(m2.coefCov) / m2.nobs) .> 2)

        #  Check computation of efficient weighting matrix in two-step procedure
        @test maximum(abs, inv(m2.spectral) .- m2.weight) < 1e-5
    end
end