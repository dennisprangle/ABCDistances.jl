using ABCDistances
using Distributions

##Set up abcinput
function sample_sumstats(pars::Array)
    success = true
    stats = [pars[1] + 0.1*randn(1), randn(1)]
    (success, stats)
end

##theta0 = [-3.0]
##srand(1)
##sobs = sample_sumstats(theta0)[2]
sobs = [0.0,0.0] ##So we get a nice graph

abcinput = ABCInput();
abcinput.prior = MvNormal(1, 100.0);
abcinput.sample_sumstats = sample_sumstats;
abcinput.abcdist = MahalanobisDiag(sobs);
abcinput.sobs = sobs;
abcinput.nsumstats = 2;

##Perform ABC-SMC
srand(20)
smcoutput1 = abcSMC(abcinput, 2000, 1000, 50000, store_init=true);
smcoutput2 = abcSMC(abcinput, 2000, 1000, 50000, adaptive=true, store_init=true);

##Look at weights
smcoutput1.abcdists[1].w
[1.0 ./ smcoutput2.abcdists[i].w for i in 1:smcoutput2.niterations]

######################################################################
##Plot simulations from each importance density and acceptance regions
######################################################################
using PyPlot
##Define plotting functions
function plot_init(out::ABCSMCOutput, i::Int32)
    ssim = out.init_sims[i]
    n = min(2500, size(ssim)[2])
    s1 = vec(ssim[1,1:n])
    s2 = vec(ssim[2,1:n])
    plot(s1, s2, ".")
end
function plot_acc(out::ABCSMCOutput, i::Int32)
    w = out.abcdists[i].w
    h = out.thresholds[i]
    ##Plot appropriate ellipse
    θ = [0:0.1:6.3]
    x = (h/w[1])*sin(θ)+sobs[1]
    y = (h/w[2])*cos(θ)+sobs[2]
    plot(x, y, lw=3)
end
nits = min(smcoutput1.niterations, smcoutput2.niterations)
PyPlot.figure(figsize=(12,12))
PyPlot.subplot(221)
for i in 1:nits
    plot_init(smcoutput1, i)
end
PyPlot.subplot(222)
for i in 1:nits
    plot_init(smcoutput2, i)
end 
PyPlot.subplot(223)
for i in 1:nits
    plot_acc(smcoutput1, i)
end
PyPlot.subplot(224)
for i in 1:nits
    plot_acc(smcoutput2, i)
end
PyPlot.savefig("normal_acc_regions.pdf")

######################################################################
##Plot RMSEs
######################################################################
PyPlot.figure(figsize=(9,3))
RMSE1 = [ sqrt(sum(smcoutput1.parameters[1,:,i].^2)) for i in 1:nits ]
RMSE2 = [ sqrt(sum(smcoutput2.parameters[1,:,i].^2)) for i in 1:nits ]   
plot(smcoutput1.cusims[1:nits], RMSE1, "r-o")
plot(smcoutput2.cusims[1:nits], RMSE2, "b-o")
xlabel("Simulations")
ylabel("RMSE")  
PyPlot.savefig("normal_RMSE.pdf")
