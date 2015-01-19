##ABCNorm defines which norm to use in an ABC algorithm.
##This file defines several concrete subtypes and two methods:
##  *  "init" initiliases the norm based on simulated summary statistics.
##     These are sometimes used to initialise weights.
##     Alternatively this method can have no effect.
##  *  "evalnorm" evaluates the norm for a particular vector of absolute values.
abstract ABCNorm

#################################
##Subtypes and methods of ABCNorm
#################################
function init(x::ABCNorm, sumstats::Array)
    ##Default is to return original norm as no initialisation required
    x
end

type Lp <: ABCNorm
    p::Number
end

function evalnorm(x::Lp, absdiff::Array)
    norm(absdiff, x.p)
end

type Euclidean <: ABCNorm
end

function evalnorm(x::Euclidean, absdiff::Array)
    norm(absdiff, 2.0)
end

type Lognorm <: ABCNorm
end

function evalnorm(x::Lognorm, absdiff::Array)
    exp(sum(log(absdiff)))
end

type MahalanobisDiag <: ABCNorm
    w::Array{Float64,1} ##Weights for each summary statistic - square root of estimated precisions
end

##Leaves w undefined
function MahalanobisDiag()
    MahalanobisDiag([])
end

function init(x::MahalanobisDiag, sumstats::Array)
    sdev = squeeze(std(sumstats, 2), 2)
    MahalanobisDiag(1.0./sdev)
end

function evalnorm(x::MahalanobisDiag, absdiff::Array)
    norm(absdiff .* x.w, 2.0)
end

type MahalanobisEmp <: ABCNorm
    Ω::Array{Float64,2} ##Weights matrix. Empirical estimate of precision used.
end

##Leaves Ω undefined
function MahalanobisEmp()
    MahalanobisEmp(eye(1))
end

function init(x::MahalanobisEmp, sumstats::Array)
    Ω = inv(cov(sumstats, vardim=2)) ##Need to deal with case where empirical covariance not invertible
    MahalanobisEmp(Ω)
end

function evalnorm(x::MahalanobisEmp, absdiff::Array)
    (absdiff' * x.Ω * absdiff)[1]
end

type MahalanobisGlasso <: ABCNorm
    Ω::Array ##Weights matrix. Empirical estimate of precision used.
end

##Leaves Ω undefined
function MahalanobisGlasso()
    MahalanobisGlasso([])
end

#= INCOMPLETE
function init(x::MahalanobisGlasso, sumstats::Array)
    Ω = ##TO DO (call python library)
    MahalanobisGlasso(Ω)
end
=#

function evalnorm(x::MahalanobisGlasso, absdiff::Array)
    (absdiff' * x.Ω * absdiff)[1]
end
