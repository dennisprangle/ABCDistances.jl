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
##ABC WITH INITIAL STATE AND σ KNOWN
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
abcinput.sobs = x0;
abcinput.nsumstats = nobs;

rejoutput = abcRejection(abcinput, 50000, 200) 
smcoutput_nonadaptive = abcSMC(abcinput, 400, 1/2, 50000);
smcoutput_adaptive = abcSMC(abcinput, 400, 1/2, 50000, adaptive=true);

abcinput.abcdist = MahalanobisEmp(x0);
smcoutput_adaptive_emp = abcSMC(abcinput, 400, 1/2, 50000, adaptive=true);

abcinput.abcdist = WeightedEuclidean(x0, "ADO");
smcoutput_ADO = abcSMC(abcinput, 400, 1/2, 50000, 50000, adaptive=true);

##Look at accepted simulations
s = smcoutput_nonadaptive
for it in (1,5,11)
    PyPlot.figure()
    PyPlot.subplot(121)
    for i in 1:20
        plot(obs_times, vec(s.sumstats[1:17,i,it]))
    end
    PyPlot.subplot(122)
    for i in 1:200
        plot(obs_times, vec(s.sumstats[18:34,i,it]))
    end
end
    
################################
##Now do ABC with initial state and σ also unknown
################################
marg_rate_prior = Uniform(-6.0,2.0)
log_sig_prior = Uniform(log(0.5),log(50.0))
prey_prior = Poisson(50)
predator_prior = Poisson(100)

type LVprior2 <: ContinuousMultivariateDistribution
end

function length(d::LVprior2)
    6
end

function _rand!{T<:Real}(d::LVprior2, x::AbstractVector{T})
    x = [rand(marg_rate_prior, 3), rand(log_sig_prior), rand(prey_prior), rand(predator_prior)]
end

function _pdf{T<:Real}(d::LVprior2, x::AbstractVector{T})
    ##n.b. initial state parameters - x[5], x[6] - are discretised to cope with SMC code adding continuous perturbations. TO DO: perturb discrete parameters more appropriately
    prod(pdf(marg_rate_prior, x[1:3])) * pdf(log_sig_prior, x[4]) * pdf(prey_prior, round(x[5])) * pdf(predator_prior, round(x[6]))
end

function sample_sumstats(pars::Array{Float64,1})
    init_state = convert(Array{Int32, 1}, round(pars[5:6]))
    (success, x) = gillespie_partial_sim(stoichiometry_LV, init_state, exp(pars[1:3]), obs_times, 100000)
    σ = exp(pars[4])
    if (success)
        stats = vec(x') + σ*randn(nobs)
    else
        stats = zeros(x)
    end
    (success, stats)
end

abcinput = ABCInput();
abcinput.prior = LVprior2();
abcinput.sample_sumstats = sample_sumstats;
abcinput.abcdist = WeightedEuclidean(x0);
abcinput.sobs = x0;
abcinput.nsumstats = nobs;

rejoutput = abcRejection(abcinput, 50000, 200)
smcoutput_nonadaptive = abcSMC(abcinput, 400, 1/2, 50000);
smcoutput_adaptive = abcSMC(abcinput, 400, 1/2, 50000, adaptive=true);
##smcoutput_nonadaptive = abcSMC(abcinput, 20000, 1/2, 1000000);
##smcoutput_adaptive = abcSMC(abcinput, 20000, 1/2, 1000000, adaptive=true);

abcinput.abcdist = WeightedEuclidean(x0, "ADO");
smcoutput_ADO = abcSMC(abcinput, 400, 1/2, 50000, adaptive=true);    

##Look at accepted simulations
s = rejoutput;
PyPlot.figure()
PyPlot.subplot(121)
for i in 1:200
    plot(obs_times, vec(s.sumstats[1:17,i]))
end
PyPlot.subplot(122)
for i in 1:200
    plot(obs_times, vec(s.sumstats[18:34,i]))
end

s = smcoutput_ADO;
for it in (1,2,3)    
    PyPlot.figure()
    PyPlot.subplot(121)
    for i in 1:20
        plot(obs_times, vec(s.sumstats[1:17,i,it]))
    end
    plot(obs_times, vec(LV_obs[1,:]), "-o")
    PyPlot.subplot(122)
    for i in 1:200
        plot(obs_times, vec(s.sumstats[18:34,i,it]))
    end
    plot(obs_times, vec(LV_obs[2,:]), "-o")
end

##NB IN THIS EXAMPLE ABC CONCENTRATES ON THE WRONG SUMMARY STATISTICS. E.G. ADO FOCUSES ON THE LATER STAGES OF THE PREDATOR POPULATION. INFORMATIVE SUMMARY STATISTICS WOULD BE VERY DESIRABLE HERE!
##This is illustrated by the following plot. This shows the length of each axis of the acceptance ellipse at each iteration.
PyPlot.figure()
for (i in 1:s.niterations)
    plot(s.thresholds[i] ./ s.abcdists[i].w)
end
