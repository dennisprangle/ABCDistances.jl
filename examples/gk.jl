using ABCDistances
using Distributions
import Distributions.length, Distributions._rand!, Distributions._pdf ##So that these can be extended
Libdl.dlopen("/usr/lib/liblapack.so.3", Libdl.RTLD_GLOBAL); ##Needed to avoid PyPlot problems on my work machine
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
quantiles = Int[1250*i for i in 1:7];
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
srand(2);
pmcoutput_alg3V = abcPMC3V(abcinput, 1000, 1/2, 1000000);
srand(2);
pmcoutput_alg5 = abcPMC5(abcinput, 1000, 1/2, 1000000);
srand(2);
pmcoutput_alg3 = abcPMC3(abcinput, 1000, 1/2, 1000000);
abcinput.abcdist = MahalanobisEmp(sobs);
srand(2);
pmcoutput_alg5_Mahalanobis = abcPMC5(abcinput, 1000, 1/2, 1000000);
abcinput.abcdist = WeightedEuclidean(sobs);
srand(2);
pmcoutput_alg4 = abcPMC4(abcinput, 1000, 1/2, 1000000);

##Plot MSEs (and also bias^2, variance)
b1 = parameter_means(pmcoutput_alg3V);
b2 = parameter_means(pmcoutput_alg5);
b3 = parameter_means(pmcoutput_alg3);
b4 = parameter_means(pmcoutput_alg5_Mahalanobis);
b5 = parameter_means(pmcoutput_alg4);
v1 = parameter_vars(pmcoutput_alg3V);
v2 = parameter_vars(pmcoutput_alg5);
v3 = parameter_vars(pmcoutput_alg3);
v4 = parameter_vars(pmcoutput_alg5_Mahalanobis);
v5 = parameter_vars(pmcoutput_alg4);
c1 = pmcoutput_alg3V.cusims ./ 1000;
c2 = pmcoutput_alg5.cusims ./ 1000;
c3 = pmcoutput_alg3.cusims ./ 1000;
c4 = pmcoutput_alg5_Mahalanobis.cusims ./ 1000;
c5 = pmcoutput_alg4.cusims ./ 1000;
PyPlot.figure(figsize=(12,8))
pnames = ("A", "B", "g", "k")
for i in 1:4
    PyPlot.subplot(220+i)
    PyPlot.plot(c3, log10(v3[i,:] .+ (b3[i,:]-theta0[i]).^2), "b-o")
    PyPlot.plot(c5, log10(v5[i,:] .+ (b5[i,:]-theta0[i]).^2), "g-^")
    PyPlot.plot(c2, log10(v2[i,:] .+ (b2[i,:]-theta0[i]).^2), "y-*")
    PyPlot.title(pnames[i])
    PyPlot.xlabel("Number of simulations (000s)")
    PyPlot.ylabel("log₁₀(MSE)")
    PyPlot.legend(["Algorithm 3","Algorithm 4", "Algorithm 5"])
end
PyPlot.tight_layout();
PyPlot.savefig("gk_mse.pdf")

PyPlot.figure()
for i in 1:4
    PyPlot.subplot(220+i)
    PyPlot.plot(c1, log10(v1[i,:]), "r-x")
    PyPlot.plot(c2, log10(v2[i,:]), "g-^")
    PyPlot.plot(c3, log10(v3[i,:]), "b-o")
    PyPlot.plot(c4, log10(v4[i,:]), "k-|")
    PyPlot.plot(c5, log10(v5[i,:]), "y-*")
    PyPlot.axis([0,maximum(vcat(c1,c2,c3,c4,c5)),-4,1]);
    PyPlot.title(pnames[i])
    PyPlot.xlabel("Number of simulations (000s)")
    PyPlot.ylabel("log₁₀(estimated variance)")
    PyPlot.legend(["Non-adaptive (alg 4)","Adaptive (alg 5)","Non-adaptive (alg 3)", "Mahalanobis", "Alg 4"])
end

PyPlot.figure()
for i in 1:4
    PyPlot.subplot(220+i)
    PyPlot.plot(c1, log10((b1[i,:]-theta0[i]).^2), "b-o")
    PyPlot.plot(c2, log10((b2[i,:]-theta0[i]).^2), "g-^")
    PyPlot.plot(c3, log10((b3[i,:]-theta0[i]).^2), "r-x")
    PyPlot.plot(c4, log10((b4[i,:]-theta0[i]).^2), "k-|")
    PyPlot.plot(c5, log10((b5[i,:]-theta0[i]).^2), "y-*")
    PyPlot.title(pnames[i])
    PyPlot.xlabel("Number of simulations (000s)")
    PyPlot.ylabel("log₁₀(bias squared)")
    PyPlot.legend(["Non-adaptive (alg 4)","Adaptive (alg 5)","Non-adaptive (alg 3)", "Mahalanobis", "Alg 4"])
end

##Posterior summaries
hcat(b3[:,end], b5[:,end], b2[:,end])' ##Rows are algs 3,4,5 of paper
sqrt(hcat(v3[:,end], v5[:,end], v2[:,end]))'

##Compute weights
w1 = pmcoutput_alg3V.abcdists[1].w;
w2 = Array(Float64, (pmcoutput_alg5.niterations, length(quantiles)));
for i in 1:pmcoutput_alg5.niterations
    w2[i,:] = pmcoutput_alg5.abcdists[i].w
end
w3 = pmcoutput_alg3.abcdists[1].w;
w5 = Array(Float64, (pmcoutput_alg4.niterations-1, length(quantiles)));
for i in 2:pmcoutput_alg4.niterations
    w5[i-1,:] = pmcoutput_alg4.abcdists[i].w
end

##Plot weights
PyPlot.figure(figsize=(12,4))
PyPlot.plot(quantiles, w3/sum(w3), "b-o")
wlast = w5[pmcoutput_alg4.niterations-1, :] ##IS -1 NEEDED?!?!
PyPlot.plot(quantiles, wlast/sum(wlast), "g-^")
wlast = w2[pmcoutput_alg5.niterations, :]
PyPlot.plot(quantiles, wlast/sum(wlast), "y-*")
##PyPlot.axis([1.0,9.0,0.0,0.35]) ##Sometimes needed to fit legend in
PyPlot.legend(["Algorithm 3", "Algorithm 4\n(last iteration)", "Algorithm 5\n(last iteration)"])
PyPlot.xlabel("Order statistic")
PyPlot.ylabel("Relative weight")
PyPlot.tight_layout();
PyPlot.savefig("gk_weights.pdf")

##Marginal posterior plots
PyPlot.figure();
samplesABC = (pmcoutput_alg3.parameters[:,:,pmcoutput_alg3.niterations], pmcoutput_alg4.parameters[:,:,pmcoutput_alg4.niterations], pmcoutput_alg5.parameters[:,:,pmcoutput_alg5.niterations]);
weightsABC = (pmcoutput_alg3.weights[:,pmcoutput_alg3.niterations], pmcoutput_alg4.weights[:,pmcoutput_alg4.niterations], pmcoutput_alg5.weights[:,pmcoutput_alg5.niterations]);
for i in 1:3 ##Loop over algorithms
    ww = weightsABC[i]
    for j in 1:4 ##Loop over parameters
        pp = samplesABC[i][j,:]
        PyPlot.subplot(140+j)
        PyPlot.plt[:hist](pp, weights=ww, normed=true, alpha=0.5)
    end
end

##Investigate best choice of alpha (as requested by reviewers)
function get_mse(pars::Array{Float64, 2}, w::Array{Float64, 1})
    mse = 0.0
    for i in 1:size(pars)[2]
        mse += sum((pars[:, i] .- theta0).^2) * w[i]
    end
    mse
end

alphas = 0.05:0.1:0.95;
MSEs_alg5 = zeros(alphas);
MSEs_alg4 = zeros(alphas);
MSEs_alg3 = zeros(alphas);
abcinput.abcdist = WeightedEuclidean(sobs);
srand(1);
for i in 1:length(alphas)
    x = abcPMC5(abcinput, 1000, alphas[i], 10^6)
    nits = x.niterations
    MSEs_alg5[i] = (nits == 0) ? Inf : get_mse(x.parameters[:,:,nits], x.weights[:,nits])
    x = abcPMC3(abcinput, 1000, alphas[i], 10^6)
    nits = x.niterations
    MSEs_alg3[i] = (nits == 0) ? Inf : get_mse(x.parameters[:,:,nits], x.weights[:,nits])
    x = abcPMC4(abcinput, 1000, alphas[i], 10^6)
    nits = x.niterations
    MSEs_alg4[i] = (nits == 0) ? Inf : get_mse(x.parameters[:,:,nits], x.weights[:,nits])    
end

PyPlot.figure();
plot(alphas, log10(MSEs_alg5), "r-x");
plot(alphas, log10(MSEs_alg3), "b-o");
plot(alphas, log10(MSEs_alg4), "g-*");
##MSE has large flat minimum around alpha in [0.3, 0.7]


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
    wv = WeightVec(s.weights[:,n])
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
    pmcoutput_alg3V = abcPMC3V(abcinput, 1000, 1/2, 1000000, silent=true)
    next!(prog)
    pmcoutput_alg5 = abcPMC5(abcinput, 1000, 1/2, 1000000, silent=true)
    next!(prog)
    pmcoutput_alg3 = abcPMC3(abcinput, 1000, 1/2, 1000000, silent=true)
    next!(prog)
    abcinput.abcdist = MahalanobisEmp(sobs)
    pmcoutput_alg5_Mahalanobis = abcPMC5(abcinput, 1000, 1/2, 1000000, silent=true)
    next!(prog)
    abcinput.abcdist = WeightedEuclidean(sobs)
    pmcoutput_alg4 = abcPMC4(abcinput, 1000, 1/2, 1000000, silent=true)
    next!(prog)
    
    trueθs[:,i] = theta0
    (squaredbiases[:,1,i], vars[:,1,i], RMSEs[:,1,i]) = getError(pmcoutput_alg3V, theta0)
    (squaredbiases[:,2,i], vars[:,2,i], RMSEs[:,2,i]) = getError(pmcoutput_alg5, theta0)
    (squaredbiases[:,3,i], vars[:,3,i], RMSEs[:,3,i]) = getError(pmcoutput_alg3, theta0)
    (squaredbiases[:,4,i], vars[:,4,i], RMSEs[:,4,i]) = getError(pmcoutput_alg5_Mahalanobis, theta0)
    (squaredbiases[:,5,i], vars[:,5,i], RMSEs[:,5,i]) = getError(pmcoutput_alg4, theta0)
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
