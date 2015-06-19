using ABCDistances
using Distributions
using Rif
using StatsBase
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
abcinput.abcdist = WeightedEuclidean(sobs, "ADO");
abcinput.sobs = sobs;
abcinput.nsumstats = length(sobs);

##Perform ABC-SMC
##(n.b. diag_perturb should be true to match SAGMB paper)
smcoutput1 = abcSMC(abcinput, 1000, 2/3, 15000, adaptive=true, diag_perturb=true);

##Inspect output weights
hcat([smcoutput1.abcdists[i].w for i in 1:smcoutput1.niterations]...)
##They do change but not dramatically

##Look at lengths of acceptance region ellipse axes
hcat([smcoutput1.thresholds[i] ./ smcoutput1.abcdists[i].w for i in 1:smcoutput1.niterations]...)

##Plot output samples
PyPlot.figure()
for it in 1:4, par in 1:3
    PyPlot.subplot(1,3,par)
    z = sample(vec(smcoutput1.parameters[par,:,it]), StatsBase.WeightVec(smcoutput1.weights[:,it]), 1000)
    PyPlot.hist(z)
end 

file=open("julia_smc_out1.dat", "w");
serialize(file, smcoutput1);
close(file);
     
##ABC WITH SEMI-AUTO SUMMARY STATISTICS
B=readdlm("semiauto_coeffs.txt");
    
##Code to sample from model and return semi-auto summary statistics using R
function sample_semiauto_sumstats(pars::Array{Float64,1})
    success = true
    rout = r_campy.simS(pars[1], pars[2], pars[3], 0.0, 1)
    features = [1, rout...]
    (success, B'*features)
end

semiauto_obs = B'*[1, sobs]
  
semiauto_abcinput = ABCInput();
semiauto_abcinput.prior = CampyPrior();
semiauto_abcinput.sample_sumstats = sample_semiauto_sumstats;
semiauto_abcinput.abcdist = MahalanobisDiag(semiauto_obs, "ADO");
semiauto_abcinput.sobs = semiauto_obs;
semiauto_abcinput.nsumstats = length(semiauto_obs);

##Perform ABC-SMC
semiauto_output = abcSMC(semiauto_abcinput, 1500, 750, 30000, adaptive=true);

##Inspect output weights
hcat([semiauto_output.abcdists[i].w for i in 1:semiauto_output.niterations]...)
##They do change but not dramatically

##Look at lengths of acceptance region ellipse axes
hcat([semiauto_output.thresholds[i] ./ semiauto_output.abcdists[i].w for i in 1:semiauto_output.niterations]...)

##Plot output samples
PyPlot.figure()
for it in 1:semiauto_output.niterations, par in 1:3
    PyPlot.subplot(1,3,par)
    z = sample(vec(semiauto_output.parameters[par,:,it]), StatsBase.WeightVec(semiauto_output.weights[:,it]), 1000)
    PyPlot.hist(z, alpha=0.5, normed=true, range=extrema(semiauto_output.parameters[par,:,1]), bins=20)
end 

PyPlot.figure()
for it in 1:semiauto_output.niterations, par in 1:3
    PyPlot.subplot(semiauto_output.niterations,3,par+3(it-1))
    PyPlot.hist(vec(semiauto_output.parameters[par,:,it]), range=extrema(semiauto_output.parameters[par,:,1]), weights=vec(semiauto_output.weights[:,it]), bins=25, normed=true)
    PyPlot.xlim(extrema(semiauto_output.parameters[par,:,1]))
    PyPlot.ylim([0.0, (0.35,0.35,3.5)[par]])
end 
