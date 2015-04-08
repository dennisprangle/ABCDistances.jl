using ABCDistances
using Distributions
import Distributions.length, Distributions._rand!, Distributions._pdf ##So that these can be extended
using PyPlot

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
abcinput.prior = GKPrior();
abcinput.sample_sumstats = sample_sumstats;
abcinput.abcdist = MahalanobisDiag(sobs);
abcinput.sobs = sobs;
abcinput.nsumstats = length(quantiles);

##Perform ABC-SMC
smcoutput1 = abcSMC(abcinput, 3000, 1000, 1000000);
smcoutput2 = abcSMC(abcinput, 3000, 1000, 1000000, adaptive=true);
abcinput.abcdist = MahalanobisDiag(sobs, "ADO")
smcoutput3 = abcSMC(abcinput, 3000, 1000, 1000000, adaptive=true);

##Plot variances
b1 = parameter_means(smcoutput1);
b2 = parameter_means(smcoutput2);
b3 = parameter_means(smcoutput3);
v1 = parameter_vars(smcoutput1);
v2 = parameter_vars(smcoutput2);
v3 = parameter_vars(smcoutput3);
c1 = smcoutput1.cusims;
c2 = smcoutput2.cusims;
c3 = smcoutput3.cusims;
PyPlot.figure(figsize=(12,12))
pnames = ("A", "B", "g", "k")
for i in 1:4
    PyPlot.subplot(220+i)
    PyPlot.plot(c1, vec(log10(v1[i,:])), "b-o")
    PyPlot.plot(c2, vec(log10(v2[i,:])), "g-o")
    ##PyPlot.plot(c3, vec(log10(v3[i,:])), "r-o") ##ADO is very similar to MAD
    PyPlot.title(pnames[i])
    PyPlot.xlabel("Number of simulations")
    PyPlot.ylabel("log10(estimated variance)")
end
PyPlot.savefig("gk_var.pdf")

PyPlot.figure()
for i in 1:4
    PyPlot.subplot(220+i)
    PyPlot.plot(c1, vec(log10((b1[i,:]-theta0[i]).^2)), "b-o")
    PyPlot.plot(c2, vec(log10((b2[i,:]-theta0[i]).^2)), "g-o")
    ##PyPlot.plot(c3, vec(log10((b3[i,:]-theta0[i]).^2)), "r-o") ##ADO is very similar to MAD    
    PyPlot.title(pnames[i])
    PyPlot.xlabel("Number of simulations")
    PyPlot.ylabel("log10(bias squared)")
end

##Compute weights
w1 = smcoutput1.abcdists[1].w;
w2 = Array(Float64, (smcoutput2.niterations, 9));
for i in 1:smcoutput2.niterations
    w2[i,:] = smcoutput2.abcdists[i].w
end

##Plot weights
PyPlot.figure(figsize=(9,3))
for i in [1,smcoutput2.niterations]
    wi = vec(w2[i,:])
    PyPlot.plot(wi/sum(wi), "-o")
end
PyPlot.xlabel("Summary statistic")
PyPlot.ylabel("Relative weight")
PyPlot.savefig("gk_weights.pdf")
