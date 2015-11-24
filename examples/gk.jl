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
    if all(0.0 .<= x .<= 10.0)
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
srand(1);
(success, sobs) = sample_sumstats(theta0)

abcinput = ABCInput();
abcinput.prior = GKPrior();
abcinput.sample_sumstats = sample_sumstats;
abcinput.abcdist = WeightedEuclidean(sobs);
abcinput.nsumstats = length(quantiles);

##Perform ABC-PMC
pmcoutput1 = abcPMC(abcinput, 1000, 1/3, 1000000);
pmcoutput2 = abcPMC(abcinput, 1000, 1/3, 1000000, adaptive=true);
pmcoutput3 = abcPMC_comparison(abcinput, 1000, 1/3, 1000000);
abcinput.abcdist = MahalanobisEmp(sobs);
pmcoutput4 = abcPMC(abcinput, 1000, 1/3, 1000000, adaptive=true);
abcinput.abcdist = WeightedEuclidean(sobs);
pmcoutput5 = abcPMC_dev(abcinput, 1000, 1/3, 1000000);

##Plot MSEs (and also bias^2, variance)
b1 = parameter_means(pmcoutput1);
b2 = parameter_means(pmcoutput2);
b3 = parameter_means(pmcoutput3);
b4 = parameter_means(pmcoutput4);
b5 = parameter_means(pmcoutput5);
v1 = parameter_vars(pmcoutput1);
v2 = parameter_vars(pmcoutput2);
v3 = parameter_vars(pmcoutput3);
v4 = parameter_vars(pmcoutput4);
v5 = parameter_vars(pmcoutput5);
c1 = pmcoutput1.cusims ./ 1000;
c2 = pmcoutput2.cusims ./ 1000;
c3 = pmcoutput3.cusims ./ 1000;
c4 = pmcoutput4.cusims ./ 1000;
c5 = pmcoutput5.cusims ./ 1000;
PyPlot.figure(figsize=(12,8))
pnames = ("A", "B", "g", "k")
for i in 1:4
    PyPlot.subplot(220+i)
    PyPlot.plot(c3, vec(log10(v3[i,:] .+ (b3[i,:]-theta0[i]).^2)), "b-o")
    PyPlot.plot(c2, vec(log10(v2[i,:] .+ (b2[i,:]-theta0[i]).^2)), "g-^")
    PyPlot.title(pnames[i])
    PyPlot.xlabel("Number of simulations (000s)")
    PyPlot.ylabel("log₁₀(MSE)")
    PyPlot.legend(["Algorithm 3","Algorithm 4"])
end
PyPlot.tight_layout();
PyPlot.savefig("gk_mse.pdf")

PyPlot.figure()
for i in 1:4
    PyPlot.subplot(220+i)
    PyPlot.plot(c1, vec(log10(v1[i,:])), "r-x")
    PyPlot.plot(c2, vec(log10(v2[i,:])), "g-^")
    PyPlot.plot(c3, vec(log10(v3[i,:])), "b-o")
    PyPlot.plot(c4, vec(log10(v4[i,:])), "k-|")
    PyPlot.plot(c5, vec(log10(v5[i,:])), "y-*")
    PyPlot.axis([0,maximum([c1,c2,c3,c4,c5]),-4,1]);
    PyPlot.title(pnames[i])
    PyPlot.xlabel("Number of simulations (000s)")
    PyPlot.ylabel("log₁₀(estimated variance)")
    PyPlot.legend(["Non-adaptive (alg 4)","Adaptive (alg 4)","Non-adaptive (alg 3)", "Mahalanobis", "Alg 5"])
end

PyPlot.figure()
for i in 1:4
    PyPlot.subplot(220+i)
    PyPlot.plot(c1, vec(log10((b1[i,:]-theta0[i]).^2)), "b-o")
    PyPlot.plot(c2, vec(log10((b2[i,:]-theta0[i]).^2)), "g-^")
    PyPlot.plot(c3, vec(log10((b3[i,:]-theta0[i]).^2)), "r-x")
    PyPlot.plot(c4, vec(log10((b4[i,:]-theta0[i]).^2)), "k-|")
    PyPlot.plot(c5, vec(log10((b5[i,:]-theta0[i]).^2)), "y-*")
    PyPlot.title(pnames[i])
    PyPlot.xlabel("Number of simulations (000s)")
    PyPlot.ylabel("log₁₀(bias squared)")
    PyPlot.legend(["Non-adaptive (alg 4)","Adaptive (alg 4)","Non-adaptive (alg 3)", "Mahalanobis", "Alg 5"])
end

##Compute weights
w1 = pmcoutput1.abcdists[1].w;
w2 = Array(Float64, (pmcoutput2.niterations, length(quantiles)));
for i in 1:pmcoutput2.niterations
    w2[i,:] = pmcoutput2.abcdists[i].w
end
w3 = pmcoutput3.abcdists[1].w;
w5 = Array(Float64, (pmcoutput5.niterations-1, length(quantiles)));
for i in 2:pmcoutput5.niterations
    w5[i-1,:] = pmcoutput5.abcdists[i].w
end

##Plot weights
PyPlot.figure(figsize=(12,4))
PyPlot.plot(quantiles, w3/sum(w3), "-o")
wlast = vec(w2[pmcoutput2.niterations, :])
PyPlot.plot(quantiles, wlast/sum(wlast), "-^")
##PyPlot.axis([1.0,9.0,0.0,0.35]) ##Sometimes needed to fit legend in
PyPlot.legend(["Algorithm 3", "Algorithm 4\n(last iteration)"])
PyPlot.xlabel("Order statistic")
PyPlot.ylabel("Relative weight")
PyPlot.tight_layout();
PyPlot.savefig("gk_weights.pdf")

wlast = vec(w5[pmcoutput5.niterations-1, :])
PyPlot.plot(quantiles, wlast/sum(wlast), "-*")
PyPlot.legend(["Algorithm 3", "Algorithm 4\n(last iteration)", "Algorithm 5"])

###############################
##ANALYSIS OF MULTIPLE DATASETS
###############################
ndatasets = 100;
trueθs = zeros((4, ndatasets));
RMSEs = zeros((4, 5, ndatasets)); ##Indices are: parameters, method, dataset
vars =  zeros((4, 5, ndatasets,));
squaredbiases = zeros((4, 5, ndatasets));

##Returns squared bias, variance and RMSE of weighted posterior sample wrt true parameters
function getError(s::ABCPMCOutput, pobs::Array{Float64, 1})
    n = s.niterations
    p = s.parameters[:,:,n]
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
prog = Progress(5*ndatasets, 1); ##Progress meter
for i in 1:ndatasets
    theta0 = rand(GKPrior())
    (success, sobs) = sample_sumstats(theta0)
    abcinput.abcdist = WeightedEuclidean(sobs)
    pmcoutput1 = abcPMC(abcinput, 1000, 1/3, 1000000, silent=true)
    next!(prog)
    pmcoutput2 = abcPMC(abcinput, 1000, 1/3, 1000000, adaptive=true, silent=true)
    next!(prog)
    pmcoutput3 = abcPMC_comparison(abcinput, 1000, 1/3, 1000000, silent=true)
    next!(prog)
    abcinput.abcdist = MahalanobisEmp(sobs)
    pmcoutput4 = abcPMC(abcinput, 1000, 1/3, 1000000, adaptive=true, silent=true)
    next!(prog)
    abcinput.abcdist = WeightedEuclidean(sobs)
    pmcoutput5 = abcPMC_dev(abcinput, 1000, 1/3, 1000000, silent=true)
    next!(prog)
    
    trueθs[:,i] = theta0
    (squaredbiases[:,1,i], vars[:,1,i], RMSEs[:,1,i]) = getError(pmcoutput1, theta0)
    (squaredbiases[:,2,i], vars[:,2,i], RMSEs[:,2,i]) = getError(pmcoutput2, theta0)
    (squaredbiases[:,3,i], vars[:,3,i], RMSEs[:,3,i]) = getError(pmcoutput3, theta0)
    (squaredbiases[:,4,i], vars[:,4,i], RMSEs[:,4,i]) = getError(pmcoutput4, theta0)
    (squaredbiases[:,5,i], vars[:,5,i], RMSEs[:,5,i]) = getError(pmcoutput5, theta0)
end

##Summarise output
mean(RMSEs, 3)
mean(vars, 3)
mean(squaredbiases, 3)

##Save output for further analysis without lengthy rerun
writedlm("gk_RMSE.txt", RMSEs)
writedlm("gk_vars.txt", vars)
writedlm("gk_bias2.txt", squaredbiases)
writedlm("gk_thetas.txt", trueθs)
