function z2gk(A::Float64, B::Float64, g::Float64, k::Float64, z::Float64, c=0.8)
    temp = exp(-g*z)
    A + B*(1.0+c*(1.0-temp)/(1.0+temp))*(1.0+z^2)^k*z    
end

function rgk(pars)
    (A,B,g,k) = pars
    z = randn()
    z2gk(A, B, g, k, z)
end

##Simulates U(0,1) order statistics specified in "orderstats" from n total sims
##orderstats should be in ascending order
##See Ripley "Stochastic Simulation" pg 98
function unif_os(orderstats::Array{Int,1}, n::Int)
    p = size(orderstats)[1]
    w = Array(Float64, p+1)
    w[1] = rand(Gamma(orderstats[1]))
    for i in 2:p
        w[i] = rand(Gamma(orderstats[i] - orderstats[i-1]))
    end
    w[p+1] = rand(Gamma(n + 1 - orderstats[p]))
    wsums = cumsum(w)
    wsums[1:p] / wsums[p+1]
end

##Efficiently simulates g&k order statistics specified in "orderstats" from n total sims
##orderstats should be in ascending order
function rgk_os(pars::Array{Float64,1}, orderstats::Array{Int,1}, n::Int)
    (A,B,g,k) = pars
    u = unif_os(orderstats, n)
    z = quantile(Normal(), u)
    map((x)->z2gk(A,B,g,k,x), z)
end
