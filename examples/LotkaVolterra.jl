using ABCDistances
Libdl.dlopen("/usr/lib/liblapack.so.3", Libdl.RTLD_GLOBAL); ##Needed to avoid PyPlot problems on my work machine
using PyPlot
using Distributions
import Distributions.length, Distributions._rand!, Distributions._pdf ##So that these can be extended

P = [ 1 0; ##Prey growth; requires 1 unit of prey
      1 1; ##Predation; requires 1 predator and 1 prey
      0 1] ##Predator death; requires 1 unit of predator
Q = [ 2 0; ##Prey growth; 2 units of prey produced
      0 2; ##Predation; 2 units of predator produced
      0 0] ##Predator death; nothing produced

stoichiometry_LV = Stoichiometry(P, Q)

##An illustration of the complete data simulation function
srand(1)
(LV_times, LV_states) = gillespie_sim(stoichiometry_LV, [50, 100], [0.5,0.0025,0.3], 100.0, 10000)

PyPlot.figure();
PyPlot.subplot(121);
plot(vec(LV_states[1,:]), vec(LV_states[2,:]))
PyPlot.title("Trajectory")
PyPlot.xlabel("Prey");
PyPlot.ylabel("Predators");
PyPlot.subplot(122);
plot(LV_times, vec(LV_states[1,:]));
plot(LV_times, vec(LV_states[2,:]));
PyPlot.legend(["Prey", "Predators"]);
PyPlot.ylim([0.,510.]); ##Stops legend blocking lines
PyPlot.title("Trace plots");
PyPlot.xlabel("Time");
PyPlot.ylabel("Population");
PyPlot.tight_layout();
PyPlot.savefig("LV_full_data.pdf");

##Simulate observed data as in Owen et al 2014 (their dataset D2), but without time 0
obs_times = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0]
state0 = [50, 100]
theta0 = [1.0, 0.005, 0.6]
srand(1)
(LV_success, LV_obs) = gillespie_partial_sim(stoichiometry_LV, state0, theta0, obs_times, 100000)
σ0 = exp(2.3) ##Scale of observation noise
nobs = 2*length(obs_times)
x0 = vec(LV_obs') + σ0*randn(nobs); ##Noisy observations
PyPlot.figure();
plot(obs_times, x0[1:16], "-o");
plot(obs_times, x0[17:32], "-o");
PyPlot.legend(["Prey", "Predators"]);
PyPlot.xlabel("Time");
PyPlot.ylabel("Noisy observation");
PyPlot.tight_layout();
PyPlot.savefig("LV_obs.pdf");

####################################
##ABC ON A SINGLE DATASET
##nb initial state and σ known
##(parameters are the log rates)
####################################
##Define prior
marg_rate_prior = Uniform(-6.0,2.0)

type LVprior <: ContinuousMultivariateDistribution
end

function length(d::LVprior)
    3
end

function _rand!{T<:Real}(d::LVprior, x::AbstractVector{T})
    x = rand(marg_rate_prior, 3)
end

function _pdf{T<:Real}(d::LVprior, x::AbstractVector{T})
    if (all(-6.0 .<= x .<= 2.0))
        return 8.0^-3
    else
        return 0.0
    end
end

##Define model
function sample_sumstats(pars::Array{Float64,1})
    (success, x) = gillespie_partial_sim(stoichiometry_LV, state0, exp(pars), obs_times, 100000)
    if (success)
        stats = vec(x') + σ0*randn(nobs)
    else
        stats = zeros(x)
    end
    (success, stats)
end

abcinput = ABCInput();
abcinput.prior = LVprior();
abcinput.sample_sumstats = sample_sumstats;
abcinput.abcdist = WeightedEuclidean(x0)
abcinput.nsumstats = nobs;

srand(1);
pmcoutput_alg3 = abcPMC3(abcinput, 200, 1/2, 50000, store_init=true);
pmcoutput_alg3V = abcPMC3V(abcinput, 200, 1/2, 50000);
pmcoutput_alg5 = abcPMC5(abcinput, 200, 1/2, 50000, store_init=true);
pmcoutput_alg4 = abcPMC4(abcinput, 200, 1/2, 50000, store_init=true);
abcinput.abcdist = MahalanobisEmp(x0)
pmcoutput_alg5_Mahalanobis = abcPMC5(abcinput, 200, 1/2, 50000, store_init=true);

##Plot weights
w1 = pmcoutput_alg3.abcdists[1].w;
w2 = pmcoutput_alg5.abcdists[pmcoutput_alg5.niterations].w;
w3 = pmcoutput_alg4.abcdists[pmcoutput_alg4.niterations].w;
PyPlot.figure(figsize=(12,6));
PyPlot.subplot(121);
plot(obs_times, w1[1:16]/sum(w1), "b-o");
plot(obs_times, w3[1:16]/sum(w3), "g-^");
plot(obs_times, w2[1:16]/sum(w2), "y-*");
PyPlot.ylim([0.,0.1])
PyPlot.xlabel("Time");
PyPlot.ylabel("Relative weight");
PyPlot.title("Prey");
PyPlot.legend(["Algorithm 3","Algorithm 4\n(last iteration)","Algorithm 5\n(last iteration)"]);
PyPlot.subplot(122);
plot(obs_times, w1[17:32]/sum(w1), "b-o");
plot(obs_times, w3[17:32]/sum(w3), "g-^");
plot(obs_times, w2[17:32]/sum(w2), "y-*");
PyPlot.ylim([0.,0.1])
PyPlot.ylabel("Relative weight");
PyPlot.xlabel("Time");
PyPlot.title("Predators");
PyPlot.legend(["Algorithm 3","Algorithm 4\n(last iteration)","Algorithm 5\n(last iteration)"]);
PyPlot.tight_layout();
PyPlot.savefig("LV_weights.pdf");

##Plot MSEs. First only those algorithms used in paper.
outputs = (pmcoutput_alg3, pmcoutput_alg4, pmcoutput_alg5);
pnames = ("Prey growth", "Predation", "Predator death");
leg_code = ("b-o", "g-^", "y-*");
PyPlot.figure(figsize=(12,6))
for i in 1:length(outputs)
    s = outputs[i]
    m = parameter_means(s)
    v = parameter_vars(s)
    c = s.cusims ./ 1000;
    for j in 1:3
        PyPlot.subplot(1, 3, j)
        PyPlot.plot(c, vec(log10(v[j,:] .+ (m[j,:]-log(theta0[j])).^2)), leg_code[i])
    end
end
for i in 1:3
    PyPlot.subplot(1, 3, i)
    PyPlot.title(pnames[i])
    PyPlot.xlabel("Number of simulations (000s)")
    PyPlot.ylabel("log₁₀(MSE)")
    PyPlot.legend(["Algorithm 3", "Algorithm 4", "Algorithm 5"])
end
PyPlot.tight_layout();
PyPlot.savefig("LV_mse.pdf");

##Add MSEs for other algorithms
outputs = (pmcoutput_alg3V, pmcoutput_alg5_Mahalanobis);
leg_code = ("r-x", "k-|");
for i in 1:length(outputs)
    s = outputs[i]
    m = parameter_means(s)
    v = parameter_vars(s)
    c = s.cusims ./ 1000;
    for j in 1:3
        PyPlot.subplot(1, 3, j)
        PyPlot.plot(c, vec(log10(v[j,:] .+ (m[j,:]-log(theta0[j])).^2)), leg_code[i])
    end
end

##Simulations used for distance initialisation

##Uncomment these rows to compare all 3 outputs
##ss_toplot = Array[pmcoutput_alg3.init_sims[1],
##                  pmcoutput_alg4.init_sims[pmcoutput_alg4.niterations],
##                  pmcoutput_alg5.init_sims[pmcoutput_alg5.niterations]];
##ss_names = ["Algorithm 3\n(first iteration)", "Algorithm 4\n(last iteration)", "Algorithm 5\n(last iteration)"];
ss_toplot = Array[pmcoutput_alg3.init_sims[1],
                  pmcoutput_alg5.init_sims[pmcoutput_alg5.niterations]];
ss_names = ["Algorithm 3\n(first iteration)", "Algorithm 5\n(last iteration)"];

PyPlot.figure(figsize=(12,12));
plotcounter = 1;
for ss in ss_toplot    
    PyPlot.subplot(length(ss_toplot), 2, plotcounter)
    for i in 1:20
        plot(obs_times, vec(ss[1:16,i]))
    end
    plot(obs_times, x0[1:16], "ko", markersize=5)
    PyPlot.ylim([0,1000])
    plotcounter += 1
    PyPlot.subplot(length(ss_toplot), 2, plotcounter)
    for i in 1:20
        plot(obs_times, vec(ss[17:32,i]))
    end
    plot(obs_times, x0[17:32], "ko", markersize=5)
    PyPlot.ylim([0,1000])
    plotcounter += 1
end
PyPlot.subplot(length(ss_toplot), 2, 1)
PyPlot.title("Prey");
PyPlot.subplot(length(ss_toplot), 2, 2)
PyPlot.title("Predators");
PyPlot.subplot(length(ss_toplot), 2, 2*length(ss_toplot)-1)
PyPlot.xlabel("Time");
PyPlot.subplot(length(ss_toplot), 2, 2*length(ss_toplot))
PyPlot.xlabel("Time");
for i in 1:length(ss_toplot)
    PyPlot.subplot(length(ss_toplot), 2, 2i-1)
    PyPlot.ylabel("Population")
    line_invis = plot((0,0),(0,0),"w-")
    PyPlot.legend([line_invis], [ss_names[i]], handlelength=0)
    PyPlot.subplot(length(ss_toplot), 2, 2i)
    PyPlot.legend([line_invis], [ss_names[i]], handlelength=0)
end
PyPlot.tight_layout();
PyPlot.savefig("LV_paths.pdf");

##Marginal posterior plots
PyPlot.figure();
samplesABC = (pmcoutput_alg3.parameters[:,:,pmcoutput_alg3.niterations], pmcoutput_alg4.parameters[:,:,pmcoutput_alg4.niterations], pmcoutput_alg5.parameters[:,:,pmcoutput_alg5.niterations]);
weightsABC = (pmcoutput_alg3.weights[:,pmcoutput_alg3.niterations], pmcoutput_alg4.weights[:,pmcoutput_alg4.niterations], pmcoutput_alg5.weights[:,pmcoutput_alg5.niterations]);
for i in 1:3 ##Loop over algorithms
    ww = weightsABC[i]
    for j in 1:3 ##Loop over parameters
        pp = vec(samplesABC[i][j,:])
        PyPlot.subplot(130+j)
        PyPlot.plt[:hist](pp, weights=ww, normed=true, alpha=0.5)
    end
end

##Posterior summaries
outputs = (pmcoutput_alg3, pmcoutput_alg4, pmcoutput_alg5);
mapreduce(p -> parameter_means(p)[:,p.niterations], hcat, outputs)' ##Posterior means
mapreduce(p -> parameter_vars(p)[:,p.niterations] |> sqrt, hcat, outputs)' ##Posterior sds

##Investigate best choice of alpha (as requested by reviewers)
function get_mse(pars::Array{Float64, 2}, w::Array{Float64, 1})
    mse = 0.0
    for i in 1:size(pars)[2]
        mse += sum((pars[:, i] .- log(theta0)).^2) * w[i]
    end
    mse
end

alphas = 0.05:0.1:0.95;
MSEs_alg5 = zeros(alphas);
MSEs_alg4 = zeros(alphas);
MSEs_alg3 = zeros(alphas);
abcinput.abcdist = WeightedEuclidean(x0);
srand(1);
for i in 1:length(alphas)
    x = abcPMC5(abcinput, 200, alphas[i], 50000)
    nits = x.niterations
    MSEs_alg5[i] = (nits == 0) ? Inf : get_mse(x.parameters[:,:,nits], x.weights[:,nits])
    x = abcPMC3(abcinput, 200, alphas[i], 50000)
    nits = x.niterations
    MSEs_alg3[i] = (nits == 0) ? Inf : get_mse(x.parameters[:,:,nits], x.weights[:,nits])
    x = abcPMC4(abcinput, 200, alphas[i], 50000)
    nits = x.niterations
    MSEs_alg4[i] = (nits == 0) ? Inf : get_mse(x.parameters[:,:,nits], x.weights[:,nits])   
end

PyPlot.figure();
plot(alphas, log10(MSEs_alg5), "r-x");
plot(alphas, log10(MSEs_alg3), "b-o");
plot(alphas, log10(MSEs_alg4), "g-*");
##Best alpha value around 0.5
