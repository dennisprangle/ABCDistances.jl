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
function init(x::ABCDistance, sumstats::Array{Float64, 2})
    ##Default is to return original distance as no initialisation required
    x
end

function init(x::ABCDistance, sumstats::Array{Float64, 2}, acceptable::Array{Bool, 1})
    ##Default is to ignore "acceptable" argument
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

function init(x::MahalanobisDiag, sumstats::Array{Float64, 2})
    (nstats, nsims) = size(sumstats)
    if (nsims == 0)
        sdev = ones(nstats)
    else
        sdev = vec(std(sumstats, 2))
    end
    return MahalanobisDiag(x.sobs, 1.0./sdev)
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

function init(x::MahalanobisEmp, sumstats::Array{Float64, 2})
    (nstats, nsims) = size(sumstats)
    if (nsims == 0)
        Ω = eye(nstats)
    else
        Ω = inv(cov(sumstats, vardim=2)) ##TO DO: need to deal with case where empirical covariance not invertible
    end
    return MahalanobisEmp(x.sobs, Ω)
end

function evaldist(x::MahalanobisEmp, s::Array{Float64, 1})
    absdiff = abs(x.sobs - s)
    (absdiff' * x.Ω * absdiff)[1]
end

type RankDist <: ABCDistance    
    sobs::Array{Float64, 1} ##Observed summaries
    N::Int32 ##How many particles to use
    s_min::Array{Float64, 1} ##Smallest acceptable value of s
    s_max::Array{Float64, 1} ##Largest acceptable value of s
end

function RankDist(sobs::Array{Float64, 1}, N::Int32)
    nstats = length(sobs)
    s_min = fill(-Inf, nstats)
    s_max = fill(Inf, nstats)
    RankDist(sobs, N, s_min, s_max)
end

function init(x::RankDist, sumstats::Array{Float64, 2})
    ##Default is that all simulations are acceptable
    (nstats, nsims) = size(sumstats)    
    acceptable = fill(true, nsims)
    init(x, sumstats, acceptable)
end

function init(x::RankDist, sumstats::Array{Float64, 2}, acceptable::Array{Bool, 1})
    (nstats, nsims) = size(sumstats)
    if (x.N >= sum(acceptable))
        s_min = fill(-Inf, nstats)
        s_max = fill(Inf, nstats)
    else
        ##marg_rank_dist is marginal signed rank distances of simulations to sobs
        s = hcat(sumstats, x.sobs) 
        marg_rank_dist = Array(Int32, (nstats, nsims))
        for i in 1:nstats
            rank = ordinalrank(vec(s[i,:]))
            marg_rank_dist[i,:] = rank[1:nsims] - rank[nsims+1]
        end
        ##max_rank_dist is an absolute rank distance of simulations to sobs
        ##It's set to Inf for simulations rejected under earlier distances
        max_rank_dist = fill(Inf, nsims)
        max_rank_dist[acceptable] = vec(maximum(abs(marg_rank_dist[:, acceptable]), 1))
        toaccept = sortperm(max_rank_dist)[1:x.N]
        sacc = sumstats[:,toaccept]
        ##Find narrowest boundaries allowing x.N acceptances (plus any ties)
        s_min = vec(minimum(sacc, 2))
        s_max = vec(maximum(sacc, 2))
        ##Expand to get widest boundaries allowing x.N acceptances (plus any ties)
        ##TO DO: Not sure this code is very efficient!
        for i in 1:nstats
            z = [vec(sumstats[i, :]), -Inf, Inf]
            s_min[i] = maximum(z[z.<s_min[i]])
            s_max[i] = minimum(z[z.>s_max[i]])
        end
        ##DEBUGGING!
        inbounds = 0
        for i in 1:nsims
            if (all(sumstats[:,i] .> s_min) & all(sumstats[:,i] .< s_max))
                inbounds += 1
            end
        end
        print("$inbounds sims are within the bounds\n")
    end
    RankDist(x.sobs, x.N, s_min, s_max)
end
    
function evaldist(x::RankDist, s::Array{Float64, 1})
    dist = 0.0 ##acceptance
    if (any(s .<= x.s_min))
        dist = 1.0 ##rejection
    elseif (any(s .>= x.s_max))
        dist = 1.0 ##rejection
    end
    return dist
end
