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
