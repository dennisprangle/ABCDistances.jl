using ABCDistances
using Distributions
import Distributions.length, Distributions._rand!, Distributions._pdf ##So that these can be extended
using PyPlot
using StatsBase
using ProgressMeter

#########################
##DEFINE MODELS AND PRIOR
#########################
##Define prior: uniform on [0,10]^4
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
quantiles = [1250*i for i in 1:7];
ndataset = 10000;

function sample_sumstats(pars::Array{Float64,1})
    success = true
    stats = rgk_os(pars, quantiles, ndataset)
    (success, stats)
end

################################################
##DETAILED ANALYSIS OF A SINGLE OBSERVED DATASET
################################################
theta0 = [3.0,1.0,1.5,0.5]
srand(1)
(success, sobs) = sample_sumstats(theta0)

abcinput = ABCInput();
abcinput.prior = GKPrior();
abcinput.sample_sumstats = sample_sumstats;
abcinput.abcdist = WeightedEuclidean(sobs);
abcinput.sobs = sobs;
abcinput.nsumstats = length(quantiles);

##Perform ABC-SMC
smcoutput1 = abcSMC(abcinput, 3000, 1/3, 1000000);
smcoutput2 = abcSMC(abcinput, 3000, 1/3, 1000000, adaptive=true);
smcoutput3 = abcSMC_comparison(abcinput, 1000, 1/3, 1000000);

##Plot variances
b1 = parameter_means(smcoutput1);
b2 = parameter_means(smcoutput2);
b3 = parameter_means(smcoutput3);
v1 = parameter_vars(smcoutput1);
v2 = parameter_vars(smcoutput2);
v3 = parameter_vars(smcoutput3);
c1 = smcoutput1.cusims ./ 1000;
c2 = smcoutput2.cusims ./ 1000;
c3 = smcoutput3.cusims ./ 1000;
PyPlot.figure(figsize=(12,12))
pnames = ("A", "B", "g", "k")
for i in 1:4
    PyPlot.subplot(220+i)
    PyPlot.plot(c1, vec(log10(v1[i,:])), "b-o")
    PyPlot.plot(c2, vec(log10(v2[i,:])), "g-^")
    PyPlot.plot(c3, vec(log10(v3[i,:])), "r-x")
    PyPlot.axis([0,maximum([c1,c2,c3]),-4,1]);
    PyPlot.title(pnames[i])
    PyPlot.xlabel("Number of simulations (000s)")
    PyPlot.ylabel("log₁₀(estimated variance)")
    PyPlot.legend(["Non-adaptive (alg 3)","Adaptive (alg 3)","Non-adaptive (alg 2)"])
end
PyPlot.tight_layout();
PyPlot.savefig("gk_var.pdf")

PyPlot.figure()
for i in 1:4
    PyPlot.subplot(220+i)
    PyPlot.plot(c1, vec(log10((b1[i,:]-theta0[i]).^2)), "b-o")
    PyPlot.plot(c2, vec(log10((b2[i,:]-theta0[i]).^2)), "g-^")
    PyPlot.plot(c3, vec(log10((b3[i,:]-theta0[i]).^2)), "r-x")
    PyPlot.title(pnames[i])
    PyPlot.xlabel("Number of simulations (000s)")
    PyPlot.ylabel("log₁₀(bias squared)")
    PyPlot.legend(["Non-adaptive (alg 3)","Adaptive (alg 3)","Non-adaptive (alg 2)"])
end

##Compute weights
w1 = smcoutput1.abcdists[1].w;
w2 = Array(Float64, (smcoutput2.niterations, length(quantiles)));
for i in 1:smcoutput2.niterations
    w2[i,:] = smcoutput2.abcdists[i].w
end
w3 = smcoutput3.abcdists[1].w;

##Plot weights
PyPlot.figure(figsize=(9,3))
wfirst = vec(w2[1,:])
PyPlot.plot(quantiles, wfirst/sum(wfirst), "-o")
wlast = vec(w2[smcoutput2.niterations, :])
PyPlot.plot(quantiles, wlast/sum(wlast), "-^")
##PyPlot.axis([1.0,9.0,0.0,0.35]) ##Sometimes needed to fit legend in
PyPlot.legend(["First iteration","Last iteration"])
PyPlot.xlabel("Order statistic (000s)")
PyPlot.ylabel("Relative weight")
PyPlot.tight_layout();
PyPlot.savefig("gk_weights.pdf")

###############################
##ANALYSIS OF MULTIPLE DATASETS
###############################
ndatasets = 5;
trueθs = zeros((4, ndatasets));
RMSEs = zeros((4, 2, ndatasets)); ## parameters x method x dataset
vars =  zeros((4, 2, ndatasets,));
squaredbiases = zeros((4, 2, ndatasets));

##Returns squared bias, variance and RMSE of weighted posterior sample wrt true parameters
function getError(s::ABCSMCOutput, pobs::Array{Float64, 1})
    n = s.niterations
    p = squeeze(s.parameters[:,:,n], 3)
    wv = WeightVec(vec(s.weights[:,n]))
    bias2 = (mean(p, wv, 2) - pobs).^2
    bias2 = vec(bias2)
    v = var(p, wv, 2)
    v = vec(v)
    rmse = sqrt(bias2 + v)
    (bias2, v, rmse)
end

abcinput = ABCInput();
abcinput.prior = GKPrior();
abcinput.sample_sumstats = sample_sumstats;
abcinput.nsumstats = length(quantiles);

srand(1);
for i in 1:ndatasets
    if i==1
        prog = Progress(2*ndatasets, 1) ##Progress meter
    end
    theta0 = rand(GKPrior())
    (success, sobs) = sample_sumstats(theta0)
    abcinput.abcdist = WeightedEuclidean(sobs)
    abcinput.sobs = sobs
    smcoutput1 = abcSMC(abcinput, 3000, 1/3, 1000000, silent=true)
    next!(prog)
    smcoutput2 = abcSMC(abcinput, 3000, 1/3, 1000000, adaptive=true, silent=true)
    next!(prog)
    trueθs[:,i] = theta0
    (squaredbiases[:,1,i], vars[:,1,i], RMSEs[:,1,i]) = getError(smcoutput1, theta0)
    (squaredbiases[:,2,i], vars[:,2,i], RMSEs[:,2,i]) = getError(smcoutput2, theta0)
end

##Summarise output
mean(RMSEs, 3)
mean(vars, 3)
mean(squaredbiases, 3)
