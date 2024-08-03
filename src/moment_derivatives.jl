#= 
Available options for df:
    exact(df) 
    forwarddiff(; step)
=#

function exact(df)
    f_gen(f) = df
    return f_gen
end

function forwarddiff(; step=1e-5)
    function f_gen(f)

        function df(b)
            # evaluate function at point b
            ff = f(b)

            # get relevant sizes
            T, M = size(ff)
            P = length(b)

            # calculate directional derivative dual
            dff = zeros(T, M, P)
            for p = 1:P
                # b change
                db = zeros(P)
                db[p] = step

                # evaluate f at b + db
                ff_fwd = f(b + db)
                dff[:, :, p] .= (ff_fwd .- ff) ./ step
            end
            return dff
        end

        return df
    end
    return f_gen
end