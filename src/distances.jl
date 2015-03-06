##An ABCDistance type defines which distance to use in an ABC algorithm.
##This file defines several concrete subtypes and two methods:
##  *  "init" initialises the distance based on simulated summary statistics.
##     These are sometimes used to initialise weights.
##     Alternatively this method can have no effect.
##     (nb infinite summary statistic values should be ignored here. They represent pre-rejected simulations.)
##  *  "evaldist" evaluates the distance for particular vector of simulated summaries.
##An ABCDistance object must be created for a particular vector of observed summaries.
##Ths is computationally convenient for some distance functions.
abstract ABCDistance

#################################
##Subtypes and methods of ABCDistance
#################################
function init(x::ABCDistance, sumstats::Array)
    ##Default is to return original distance as no initialisation required
    x
end

type Lp <: ABCDistance
    sobs::Array{Float64,1}
    p::Float64
end

function evaldist(x::Lp, s::Array{Float64,1})
    absdiff = abs(x.sobs - s)
    norm(absdiff, x.p)
end

type Euclidean <: ABCDistance
    sobs::Array{Float64,1}
end

function evaldist(x::Euclidean, s::Array{Float64,1})
    absdiff = abs(x.sobs - s)
    norm(absdiff, 2.0)
end

type Logdist <: ABCDistance
    sobs::Array{Float64,1}
end

function evaldist(x::Logdist, s::Array{Float64,1})
    absdiff = abs(x.sobs - s)
    exp(sum(log(absdiff)))
end

type MahalanobisDiag <: ABCDistance
    sobs::Array{Float64,1}
    w::Array{Float64,1} ##Weights for each summary statistic - square root of estimated precisions
end

##Leaves w undefined
function MahalanobisDiag(sobs::Array{Float64, 1})
    MahalanobisDiag(sobs, [])
end

##Handles initialisation in case where some summary statistics should be ignored
function init_inf(x::MahalanobisDiag, sumstats::Array{Float64, 2})
    todrop = vec(isinf(sumstats[1,:]))
    if (all(todrop))
        sdev = fill(1, size(sumstats)[1])
    else
        s = sumstats[:,!todrop]
        sdev = vec(std(s, 2))
    end
    MahalanobisDiag(x.sobs, 1.0./sdev)
end

function init(x::MahalanobisDiag, sumstats::Array{Float64, 2})
    if (any(isinf(sumstats)))
        return init_inf(x, sumstats)
    else
        sdev = vec(std(sumstats, 2))
        return MahalanobisDiag(x.sobs, 1.0./sdev)
    end
end

function evaldist(x::MahalanobisDiag, s::Array{Float64, 1})
    absdiff = abs(x.sobs - s)
    norm(absdiff .* x.w, 2.0)
end

type MahalanobisEmp <: ABCDistance
    sobs::Array{Float64,1}
    Ω::Array{Float64,2} ##Weights matrix. Empirical estimate of precision used.
end

##Leaves Ω undefined
function MahalanobisEmp(sobs::Array{Float64, 1})
    MahalanobisEmp(sobs, eye(1))
end

##Handles initialisation in case where some summary statistics should be ignored
function init_inf(x::MahalanobisEmp, sumstats::Array{Float64, 2})
    todrop = vec(isinf(sumstats[1,:]))
    if (all(todrop))
        Ω = eye(size(sumstats)[1])
    else
        s = sumstats[:,!todrop]
        Ω = inv(cov(s, vardim=2))
    end
    MahalanobisEmp(x.sobs, Ω)
end

function init(x::MahalanobisEmp, sumstats::Array{Float64, 2})
    if (any(isinf(sumstats)))
        return init_inf(x, sumstats)
    else
        Ω = inv(cov(sumstats, vardim=2)) ##Need to deal with case where empirical covariance not invertible
        return MahalanobisEmp(x.sobs, Ω)
    end
end

function evaldist(x::MahalanobisEmp, s::Array{Float64, 1})
    absdiff = abs(x.sobs - s)
    (absdiff' * x.Ω * absdiff)[1]
end

type MahalanobisNP <: ABCDistance
    sobs::Array{Float64, 1} ##Observed summaries
    zobs::Array{Float64, 1} ##Observed summaries transformed to normal variates
    sumstats::Array{Float64, 2} ##Sorted summaries sampled via importance dist. sumstats[i,j] is ith largest value for summary j.
end

function MahalanobisNP(sobs::Array{Float64, 1})
    MahalanobisNP(sobs, [0.5], eye(1))
end

function init(x::MahalanobisNP, sumstats::Array{Float64, 2})
    Sin = hcat(sumstats, x.sobs) ##Include the observations as well
    (nstats, nsims) = size(Sin)
    Sout = zeros(nsims, nstats)
    for i in 1:nstats
        Sout[:,i] = sort(vec(Sin[i,:]))
    end
    zobs = [pw_cdf(Sout[:,i], x.sobs[i]) for i in 1:nstats]
    zobs = quantile(Normal(), zobs)
    MahalanobisNP(x.sobs, zobs, Sout)
end

function evaldist(x::MahalanobisNP, s::Array{Float64, 1})
    z = [pw_cdf(x.sumstats[:,i], s[i]) for i in 1:length(s)]
    z = quantile(Normal(), z)
    absdiff = abs(x.zobs - z)
    norm(absdiff, 2.0)
end

##Piece-wise linear cdf estimate for x from sorted data a
function pw_cdf(a::Array{Float64, 1}, x::Float64)
    n = length(a)
    i = searchsortedfirst(a, x)
    ##(al, ar) are values used for interpolation
    ##(cl, cr) are cdf values at these points
    if (i==1)
        al = a[1]
        ar = a[2]
        (cl, cr) = [1,2]/(n+1)
    elseif (i==n+1)
        al = a[n-1]
        ar = a[n]
        (cl, cr) = [n-1,n]/(n+1)
    else
        al = a[i-1]
        ar = a[i]
        (cl, cr) = [i-1,i]/(n+1)
    end
    y = cl + (cr-cl)*(x-al)/(ar-al)
    max(0.0, min(y,1.0))    
end
