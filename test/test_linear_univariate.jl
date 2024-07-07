
@testset "Univariate Regression" begin

    @testset "Argument Consistency" begin
        y = zeros(100)
        x = zeros(100, 2)
        z = zeros(100, 3)

        @test_throws ArgumentError regIV(zeros(0), x, z)
        @test_throws ArgumentError regIV(y, zeros(50, 2), z)
        @test_throws ArgumentError regIV(y, x, z; weight=zeros(3, 2))
        @test_throws ArgumentError regIV(y, x, z; weight=zeros(2, 2))
    end

    @testset "Solution Precision: Test 1" begin
        # Generate random data
        n = 10000
        x = randn(n)
        z = x + 0.1 * randn(n)  # Slightly noisy version of x
        e = 0.1 * randn(n)
        y = 2.0 .+ 3.0 .* x .+ e  # y = 2 + 3x + e

        # Run the regression models
        ols_reg = regOLS(y, x; spectral_model=white())
        iv_reg = regIV(y, x, z; spectral_model=nw(3))

        # Check that the estimated coefficients are close to the true values
        @test abs(ols_reg.intercept - 2.0) < 0.1
        @test abs(ols_reg.coef[1] - 3.0) < 0.1
        @test abs(iv_reg.intercept - 2.0) < 0.1
        @test abs(iv_reg.coef[1] - 3.0) < 0.1

        # Check that the estimated coefficients are statistically significant
        @test abs(ols_reg.tStat[1]) > 2
        @test abs(iv_reg.tStat[1]) > 2

        # Check that the estimated residual variance is close to true value
        @test abs(ols_reg.resStDev - 0.1) < 0.02
        @test abs(iv_reg.resStDev - 0.1) < 0.02

        # Check that the residuals are small
        @test maximum(abs, ols_reg.res) < 1.0
        @test maximum(abs, iv_reg.res) < 1.0
    end

    @testset "Solution Precision: Test 2" begin
        # Generate random data
        n = 10000
        x = randn(n, 2)
        z = x + 0.3 * randn(n, 2)  # Slightly noisy version of x
        e = 0.1 * randn(n)
        y = 1.0 .+ 2.0 .* x[:, 1] .+ 3.0 .* x[:, 2] .+ e

        # Run the regression models
        ols_reg = regOLS(y, x; spectral_model=hh(6))
        iv_reg = regIV(y, x, z; spectral_model=nw(20))

        # Check that the estimated coefficients are close to the true values
        @test abs(ols_reg.intercept - 1.0) < 0.1
        @test abs(ols_reg.coef[1] - 2.0) < 0.1
        @test abs(ols_reg.coef[2] - 3.0) < 0.1
        @test abs(iv_reg.intercept - 1.0) < 0.1
        @test abs(iv_reg.coef[1] - 2.0) < 0.1
        @test abs(iv_reg.coef[2] - 3.0) < 0.1

        # Check that the estimated coefficients are statistically significant
        @test abs(ols_reg.tStat[1]) > 2
        @test abs(ols_reg.tStat[2]) > 2
        @test abs(iv_reg.tStat[1]) > 2
        @test abs(iv_reg.tStat[2]) > 2

        # Check that the estimated residual variance is close to true value
        @test abs(ols_reg.resStDev - 0.1) < 0.02
        @test abs(iv_reg.resStDev - 0.1) < 0.02

        # Check that the residuals are small
        @test maximum(abs, ols_reg.res) < 1.0
        @test maximum(abs, iv_reg.res) < 1.0
    end

    @testset "Rank and Order Conditions" begin
        n = 100
        x = randn(n, 2)
        z = [nullspace(x')[:, 1:2] x[:, 1]]
        e = 0.1 * randn(n)
        y = 2.0 .+ 3.0 .* x[:, 1] .+ 4.0 .* x[:, 2] .+ e

        @test_throws ArgumentError regIV(y, x, z[:, 1])
        @test_throws ArgumentError regIV(y, x, z)
    end
end