
@testset "Instrumental Variables Regression Tests" begin
    # Generate some random data for testing
    n = 10000
    x = randn(n)
    z = x + 0.1 * randn(n)  # Slightly noisy version of x
    e = 0.1 * randn(n)
    y = 2.0 .+ 3.0 .* x .+ e  # y = 2 + 3x + e

    # Run the IV regression
    iv_reg = regIV(y, x, z)

    # Check that the estimated coefficients are close to the true values
    @test abs(iv_reg.intercept - 2.0) < 0.1  # Intercept
    @test abs(iv_reg.coef[1] - 3.0) < 0.1  # Slope

    # Check that the estimated coefficients are statistically significant
    @test abs(iv_reg.tStat[1]) > 2

    # Check that the estimated residual variance is close to true value
    @test abs(iv_reg.resStDev - 0.1) < 0.02

    # Check that the residuals are small
    @test maximum(abs, iv_reg.res) < 1.0
end

@testset "Ordinary Least Squares Regression Tests" begin
    # Generate some random data for testing
    n = 10000
    x = randn(n)
    e = 0.1 * randn(n)
    y = 2.0 .+ 3.0 .* x .+ e  # y = 2 + 3x + e

    # Run the OLS regression
    ols_reg = regOLS(y, x)

    # Check that the estimated coefficients are close to the true values
    @test abs(ols_reg.intercept - 2.0) < 0.1  # Intercept
    @test abs(ols_reg.coef[1] - 3.0) < 0.1  # Slope

    # Check that the estimated coefficients are statistically significant
    @test abs(ols_reg.tStat[1]) > 2

    # Check that the estimated residual variance is close to true value
    @test abs(ols_reg.resStDev - 0.1) < 0.02

    # Check that the residuals are small
    @test maximum(abs, ols_reg.res) < 1.0
end