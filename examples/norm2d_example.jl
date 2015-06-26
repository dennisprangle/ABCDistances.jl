using ABCDistances
using Distributions
using PyPlot

######################################################################
##Define plotting functions
######################################################################
plot_cols = ("b", "g", "r", "c", "m", "y", "k");
function plot_init(out::ABCSMCOutput, i::Int32)
    ssim = out.init_sims[i]
    n = min(2500, size(ssim)[2])
    s1 = vec(ssim[1,1:n])
    s2 = vec(ssim[2,1:n])
    plot(s1, s2, ".", color=plot_cols[i])
end
function plot_acc(out::ABCSMCOutput, i::Int32)
    w = out.abcdists[i].w
    h = out.thresholds[i]
    ##Plot appropriate ellipse
    θ = [0:0.1:6.3]
    x = (h/w[1])*sin(θ)+sobs[1]
    y = (h/w[2])*cos(θ)+sobs[2]
    plot(x, y, lw=3, color=plot_cols[i])
end

######################################################################
##Run ABC
######################################################################
##Set up abcinput
function sample_sumstats(pars::Array)
    success = true
    stats = [pars[1] + 0.1*randn(1), randn(1)]
    (success, stats)
end

sobs = [0.0,0.0];

abcinput = ABCInput();
abcinput.prior = MvNormal(1, 100.0);
abcinput.sample_sumstats = sample_sumstats;
abcinput.abcdist = WeightedEuclidean(sobs);
abcinput.sobs = sobs;
abcinput.nsumstats = 2;

##Perform ABC-SMC
srand(20);
smcoutput_adapt = abcSMC(abcinput, 1000, 1/2, 50000, adaptive=true, store_init=true);
srand(20);
abcinput.abcdist = smcoutput_adapt.abcdists[1];
smcoutput_noadapt = abcSMC_comparison(abcinput, 1000, 1/2, 50000, h1=smcoutput_adapt.thresholds[1], initialise_dist=false, store_init=true);
srand(20);
abcinput.abcdist = WeightedEuclidean(sobs);
smcoutput_noadapt2 = abcSMC(abcinput, 1000, 1/2, 50000, adaptive=false, store_init=true);

##Look at weights
smcoutput_noadapt.abcdists[1].w
smcoutput_noadapt2.abcdists[1].w
[smcoutput_adapt.abcdists[i].w for i in 1:smcoutput_adapt.niterations]

##Plot simulations from each importance density and acceptance regions
nits = min(smcoutput_noadapt.niterations, smcoutput_adapt.niterations)
PyPlot.figure(figsize=(12,12))
PyPlot.subplot(221)
for i in 1:nits
    plot_init(smcoutput_noadapt, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$")
PyPlot.title("Non-adaptive simulations")
PyPlot.subplot(222)
for i in 1:nits
    plot_init(smcoutput_adapt, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$") 
PyPlot.title("Adaptive simulations")
PyPlot.subplot(223)
for i in 1:nits
    plot_acc(smcoutput_noadapt, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$")
PyPlot.title("Non-adaptive acceptance regions")
PyPlot.subplot(224)
for i in 1:nits
    plot_acc(smcoutput_adapt, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$")
PyPlot.title("Adaptive acceptance regions")
PyPlot.tight_layout()
PyPlot.savefig("normal_acc_regions.pdf")

##Plot RMSEs
PyPlot.figure(figsize=(9,3))
RMSE1 = [ sqrt(sum(smcoutput_noadapt.parameters[1,:,i].^2)) for i in 1:nits ]
RMSE2 = [ sqrt(sum(smcoutput_adapt.parameters[1,:,i].^2)) for i in 1:nits ]   
plot(smcoutput_noadapt.cusims[1:nits], RMSE1, "r-o")
plot(smcoutput_adapt.cusims[1:nits], RMSE2, "b-x")
xlabel("Simulations")
ylabel("RMSE")
legend(["Algorithm 2 (non-adaptive)","Algorithm 3 (adaptive)"])
PyPlot.tight_layout()
PyPlot.savefig("normal_RMSE.pdf")
