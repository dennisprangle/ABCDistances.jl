using ABCDistances
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
(LV_times, LV_states) = gillespie_sim(stoichiometry_LV, [200, 300], [0.5,0.0025,0.3], 100.0, 10000)

PyPlot.figure()
PyPlot.subplot(121)
plot(LV_times, vec(LV_states[1,:]))
plot(LV_times, vec(LV_states[2,:]))
PyPlot.subplot(122)
plot(vec(LV_states[1,:]), vec(LV_states[2,:]))

##Simulate observed data as in Owen et al 2014 (their dataset D2)
obs_times = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0]
state0 = [50, 100]
theta0 = [1.0, 0.005, 0.6]
srand(1)
(LV_success, LV_obs) = gillespie_partial_sim(stoichiometry_LV, state0, theta0, obs_times, 100000)
PyPlot.figure()
plot(obs_times, vec(LV_obs[1,:]), "-o")
plot(obs_times, vec(LV_obs[2,:]), "-o")
σ0 = exp(2.3) ##Scale of observation noise
nobs = 2*length(obs_times)
x0 = vec(LV_obs') + σ0*randn(nobs); ##Noisy observations

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
    ##TO DO: IS UPPER LIMIT ON SIMS EVER REACHED?
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
smcoutput_nonadaptive = abcSMC_comparison(abcinput, 200, 1/2, 50000, store_init=true);
smcoutput_nonadaptive2 = abcSMC(abcinput, 200, 1/2, 50000);
smcoutput_adaptive = abcSMC(abcinput, 200, 1/2, 50000, adaptive=true, store_init=true);
abcinput.abcdist = MahalanobisEmp(x0)
smcoutput_Mahalanobis = abcSMC(abcinput, 200, 1/2, 50000, adaptive=true, store_init=true);

##Plot weights
w1 = smcoutput_nonadaptive.abcdists[1].w;
w2 = smcoutput_adaptive.abcdists[smcoutput_adaptive.niterations].w;
PyPlot.figure(figsize=(12,6));
PyPlot.subplot(121);
plot(obs_times, w1[1:17]/sum(w1), "b-o");
plot(obs_times, w2[1:17]/sum(w2), "g-^");
PyPlot.ylim([0.,0.18])
PyPlot.xlabel("Time");
PyPlot.ylabel("Relative weight");
PyPlot.title("Prey");
PyPlot.legend(["Algorithm 3","Algorithm 4\n(last iteration)"]);
PyPlot.subplot(122);
plot(obs_times, w1[18:34]/sum(w2), "b-o");
plot(obs_times, w2[18:34]/sum(w2), "g-^");
PyPlot.ylim([0.,0.18])
PyPlot.ylabel("Relative weight");
PyPlot.xlabel("Time");
PyPlot.title("Predators");
PyPlot.legend(["Algorithm 3","Algorithm 4\n(last iteration)"]);
PyPlot.tight_layout();
PyPlot.savefig("LV_weights.pdf");

##Plot MSEs. First only those algorithms used in paper.
outputs = (smcoutput_nonadaptive, smcoutput_adaptive);
pnames = ("Prey growth", "Predation", "Predator death");
leg_code = ("b-o", "g-^");
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
    PyPlot.legend(["Algorithm 3", "Algorithm 4"])
end
PyPlot.tight_layout();
PyPlot.savefig("LV_mse.pdf");

##Add MSEs for other algorithms
outputs = (smcoutput_nonadaptive2, smcoutput_Mahalanobis);
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
ss_toplot = Array[smcoutput_nonadaptive.init_sims[1],
                  smcoutput_adaptive.init_sims[smcoutput_adaptive.niterations]];
ss_names = ["Algorithm 3\n(first iteration)", "Algorithm 4\n(last iteration)"];             
PyPlot.figure(figsize=(12,12));
plotcounter = 1;
for ss in ss_toplot    
    PyPlot.subplot(length(ss_toplot), 2, plotcounter)
    for i in 1:20
        plot(obs_times, vec(ss[1:17,i]))
    end
    plot(obs_times, x0[1:17], "ko", markersize=5)
    PyPlot.ylim([0,800])
    plotcounter += 1
    PyPlot.subplot(length(ss_toplot), 2, plotcounter)
    for i in 1:20
        plot(obs_times, vec(ss[18:34,i]))
    end
    plot(obs_times, x0[18:34], "ko", markersize=5)
    PyPlot.ylim([0,800])
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
