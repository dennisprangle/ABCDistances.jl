using ABCDistances
using Distributions
using Rif
import Distributions.length, Distributions._rand!, Distributions._logpdf ##So that these can be extended
using PyPlot

##Define prior
type CampyPrior <: ContinuousMultivariateDistribution
end

function length(d::CampyPrior)
    3
end

function _rand!{T<:Real}(d::CampyPrior, x::AbstractVector{T})
    logcross::Float64 = rand(TruncatedNormal(0.27,  1.873, -Inf, log(25.0)))
    logtract::Float64 = rand(TruncatedNormal(1.513, 1.956, log(1/1000), 25.0)) ##ms crashes outside these bounds
    logmut::Float64   = rand(         Normal(2.617, 0.2683))
    [logcross, logtract, logmut]
end

function _logpdf{T<:Real}(d::CampyPrior, x::AbstractVector{T})
    l::Float64 = logpdf(TruncatedNormal(0.27,  1.873, -Inf, log(25.0)), x[1])
    l         += logpdf(TruncatedNormal(1.513, 1.956, log(1/1000), 25.0), x[2])
    l         += logpdf(Normal(2.617, 0.2683), x[3])
    l
end

##Set up Rif interface to R and initialise appropriately
Rif.initr();
r_campy = Rif.importr("campy");
r_campy.setPath(["/home/dennis"]);

##Code to sample from model using R
function sample_sumstats(pars::Array{Float64,1})
    success = true
    rout = r_campy.simS(pars[1], pars[2], pars[3], 0.0, 1)
    stats = [rout...]
    (success, stats)
end

##Observed sum stats (calculated in R)
sobs = [0.9310435, 10.0, 6.1, 296.0, 53.5853465, 5.5660606, 36.0, 18.0, 2.7777778, 16.2857143, 5.5714286, 35.1428571, 106.1428571, 6.2384954, 0.6336428];

abcinput = ABCInput();
abcinput.prior = CampyPrior();
abcinput.sample_sumstats = sample_sumstats;
abcinput.abcdist = MahalanobisDiag(sobs, "ADO");
abcinput.sobs = sobs;
abcinput.nsumstats = length(sobs);

##Perform ABC-SMC
##(n.b. diag_perturb should be true to match SAGMB paper)
smcoutput1 = abcSMC(abcinput, 1500, 1000, 15000, adaptive=true, diag_perturb=true);

##Inspect output weights
[smcoutput1.abcdists[i].w for i in 1:smcoutput1.niterations]

##Plot output sample
PyPlot.figure()
for it in 1:2, par in 1:3
    PyPlot.subplot(230+3*(it-1)+par)
    PyPlot.hist(vec(smcoutput1.parameters[par,:,it]))
end
