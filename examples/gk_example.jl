using ABCDistances
using Distributions
import Distributions.length, Distributions._rand!, Distributions._pdf ##So that these can be extended

##Define prior
type GKPrior <: ContinuousMultivariateDistribution
end

function length(d::GKPrior)
    4
end

function _rand!{T<:Real}(d::GKPrior, x::AbstractVector{T})
    x = 10.0*rand(4)
end

function _pdf{T<:Real}(d::GKPrior, x::AbstractVector{T})
    if (all(0.0 .<= x .<= 10.0))
        return 0.0001
    else
        return 0.0
    end
end

##Define model and summary statistics
quantiles = [1000,2000,3000,4000,5000,6000,7000,8000,9000]
ndataset = 10000

function sample_sumstats(pars::Array{Float64,1})
    success = true
    stats = rgk_os(pars, quantiles, ndataset)
    (success, stats)
end

theta0 = [3.0,1.0,1.5,0.5]
srand(1)
(success, sobs) = sample_sumstats(theta0)

abcinput = ABCInput();
abcinput.prior = GKPrior()
abcinput.sample_sumstats = sample_sumstats
abcinput.abcdist = MahalanobisEmp(sobs)
abcinput.sobs = sobs
abcinput.nsumstats = length(quantiles)

##Perform ABC rejection
abcoutput = abcRejection(abcinput, 10000, 200)
##abcoutput = abcRejection(abcinput, 10000, 0.3)

##Plot results
using PyPlot
pars = abcoutput.parameters
PyPlot.figure()
for i in 1:4
    PyPlot.subplot(220+i)
    PyPlot.hist(vec(pars[i,:]))
end
##What is fitted precision matrix?
abcoutput.abcdist.Ω

##Perform ABC-SMC
abcinput.abcdist = MahalanobisDiag(sobs)
smcoutput1 = abcSMC(abcinput, 800, 200, 2500000);
smcoutput2 = abcSMC(abcinput, 800, 200, 2500000, adaptive=true);
abcinput.abcdist = MahalanobisEmp(sobs)
smcoutput3 = abcSMC(abcinput, 800, 200, 2500000, adaptive=true);

##Plot variances
PyPlot.figure()
v1 = parameter_vars(smcoutput1);
v2 = parameter_vars(smcoutput2);
c1 = smcoutput1.cusims;
c2 = smcoutput2.cusims;
PyPlot.figure()
for i in 1:4
    PyPlot.subplot(220+i)
    PyPlot.plot(log(c1), vec(log(v1[i,:])))
    PyPlot.plot(log(c2), vec(log(v2[i,:])))
end

##Compute weights
w1 = smcoutput1.abcdists[1].w;
w2 = Array(Float64, (smcoutput2.niterations, 9));
for i in 1:smcoutput2.niterations
    w2[i,:] = smcoutput2.abcdists[i].w
end

##Plot weights
PyPlot.figure()
PyPlot.subplot(211)
PyPlot.plot(w1/sum(w1))
PyPlot.subplot(212)
for i in 1:smcoutput2.niterations
    wi = vec(w2[i,:])
    PyPlot.plot(wi/sum(wi))
end
