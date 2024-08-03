#= 

Available options for spectral:
    preset(S)
    nw(k)
    hh(k) 
    white() 

=#

# TODO: use preset to build 2SLS

function preset(S)
    S_gen(f, b) = S
    return S_gen
end

demean(X::Vector) = X .- mean(X)
demean(X::Matrix) = mapslices(demean, X, dims=1)

function S_estimator(k::Int64, w::Function)
    function S_gen(f, b)
        F = demean(f(b))
        T, M = size(F)
        S = zeros(M, M)
        for j in range(-k, k, step=1)
            wei = w(j, k)
            (wei == 0) && continue

            ja = abs(j)
            if j > 0
                v_r = F[ja+1:end, :] # enters multiplication as row vector
                v_c = F[1:end-ja, :]
            elseif j < 0
                v_r = F[1:end-ja, :] # enters multiplication as row vector
                v_c = F[ja+1:end, :]
            elseif j == 0
                v_r = F
                v_c = F
            end
            av = (v_r' * v_c) / (T - ja)
            S += wei * av
        end
        return S
    end
    return S_gen
end

nw(k::Int64) = S_estimator(k, (j, k) -> (k - abs(j)) / k)
hh(k::Int64) = S_estimator(k, (j, k) -> 1)
white() = S_estimator(0, (j, k) -> 1)

