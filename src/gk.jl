##TO DO: integrate with style of Distributions package
##       adds fast quantiles sampler

function z2gk(A, B, g, k, z, c=0.8)
    temp = exp(-g*z)
    A + B*(1.0+c*(1.0-temp)/(1.0+temp))*(1.0+z^2)^k*z    
end

function rgk(pars)
    (A,B,g,k) = pars
    z = randn()
    z2gk(A, B, g, k, z)
end
