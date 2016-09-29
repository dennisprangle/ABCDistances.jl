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
"
`ABCDistance` objects are constructed specifying the observed summaries `sobs`.
Some types require further information which is initialised during an ABC algorithm.
This is set up by the `init` method.
Its arguments are

* `x` The `ABCDistance` object
* `sumstats` All the simulated summaries from the current iteration (including rejections). This is a `Array{Float64, 2}` whose columns are the simulations.
* `parameters` The parameters corresponding to the `sumstats` simulations. This is also a `Array{Float64, 2}` whose columns are the simulations.

`abcRejection` always calls this method.
So does `abcPMC` on the first iteration, or all iterations for `adaptive=true`.
`abcPMC_comparison` calls this on the first iteration if `initialise_dist=true` or otherwise not at all.
"
function init(x::ABCDistance, sumstats::Array{Float64, 2})
    ##Default is to return original distance as no initialisation required
    x
end

function init(x::ABCDistance, sumstats::Array{Float64, 2}, parameters::Array{Float64, 2})
    ##Default is to ignore parameters, as these are often not needed
    init(x, sumstats)
end


type Lp <: ABCDistance
    sobs::Array{Float64,1}
    p::Float64
end

function evaldist(x::Lp, s::Array{Float64,1})
    absdiff = abs(x.sobs - s)
    norm(absdiff, x.p)
end

"
Euclidean distance. Its constructor is `Euclidean(sobs)`.
"
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

"""
This is weighted Euclidean distance with summary statistics weighted by scalars - equation (3) in the paper.
It has several constructors.
  * `WeightedEuclidean(sobs, w, scale_type)`
  Here `w` is the weights for each summary and `scale_type` is how to initialise the scale estimate. Possible values for the latter include `"sd"` and `"MAD"` (standard deviation and median absolute deviation).
  * `WeightedEuclidean(sobs, scale_type)`
  Here the weights are left undefined until initialisation of the distance takes place.
  * `WeightedEuclidean(sobs)`
  Equivalent to `WeightedEuclidean(sons, "MAD")`
"""
type WeightedEuclidean <: ABCDistance
    sobs::Array{Float64,1}
    w::Array{Float64,1} ##Weights for each summary statistic - square root of estimated precisions
    scale_type::AbstractString ##Whether to initialise scale using "MAD", "sd" or "ADO"

    function WeightedEuclidean(sobs::Array{Float64,1}, w::Array{Float64,1}, scale_type::AbstractString)
        if (scale_type ∉ ("MAD", "sd", "sdreg", "ADO"))
            error("scale_type must be MAD sd sdreg or ADO")
        end
        new(sobs, w, scale_type)
    end
end

function WeightedEuclidean(sobs::Array{Float64, 1}, scale_type::AbstractString)
    WeightedEuclidean(sobs, Array(Float64,0), scale_type)
end

function WeightedEuclidean(sobs::Array{Float64, 1})
    WeightedEuclidean(sobs, Array(Float64,0), "MAD")
end

function init(x::WeightedEuclidean, sumstats::Array{Float64, 2}, parameters::Array{Float64, 2})
    (nstats, nsims) = size(sumstats)
    if (nsims == 0)
        sig = ones(nstats)
    elseif x.scale_type=="MAD"
        sig = Float64[MAD(sumstats[i,:]) for i in 1:nstats]
    elseif x.scale_type=="sd"
        sig = Float64[std(sumstats[i,:]) for i in 1:nstats]
    elseif x.scale_type=="sdreg"
        P = parameters'
        sig = Array(Float64, nstats)
        for i in 1:nstats
            y = sumstats[i,:]
            beta = linreg(P, y)
            res = y - beta[1] - P * beta[2:end]
            sig[i] = std(res)
        end
    elseif x.scale_type=="ADO"
        sig = [ADO(sumstats[i,:], x.sobs[i]) for i in 1:nstats]
    end
    return WeightedEuclidean(x.sobs, 1.0./sig, x.scale_type)
end

##Median absolute deviation
function MAD(x::Array{Float64, 1})
    median(abs(x - median(x)))
end

##Absolute deviation to observations
function ADO(x::Array{Float64, 1}, obs::Float64)
    median(abs(x - obs))
end

function evaldist(x::WeightedEuclidean, s::Array{Float64, 1})
    absdiff = abs(x.sobs - s)
    norm(absdiff .* x.w, 2.0)
end

"
This is weighted Euclidean distance with summary statistics weighted by a matrix i.e. d(x,y)=(x-y)' W (x-y). (This can often be viewed as an empirical estimate of Mahalanobis distance.)
It has two constructors.
  * `MahalanobisEmp(sobs, Ω)`
  Where `Ω` is used as the matrix W.
  * `MahalanobisEmp(sobs)`
  Here the weights are left undefined until initialisation of the distance takes place.
"          
type MahalanobisEmp <: ABCDistance
    sobs::Array{Float64,1}
    Ω::Array{Float64,2} ##Weights matrix. Empirical estimate of precision used.
end

function MahalanobisEmp(sobs::Array{Float64, 1})
    MahalanobisEmp(sobs, eye(1))
end

function init(x::MahalanobisEmp, sumstats::Array{Float64, 2})
    (nstats, nsims) = size(sumstats)
    if (nsims == 0)
        Ω = eye(nstats)
    else
        Ω = inv(cov(sumstats, 2)) ##TO DO: need to deal with case where empirical covariance not invertible
    end
    return MahalanobisEmp(x.sobs, Ω)
end

function evaldist(x::MahalanobisEmp, s::Array{Float64, 1})
    absdiff = abs(x.sobs - s)
    (absdiff' * x.Ω * absdiff)[1]
end
