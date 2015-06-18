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
##EXAMPLE 1: one informative summary statistic
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
smcoutput1 = abcSMC(abcinput, 1000, 1/2, 50000, store_init=true);
srand(20);
smcoutput2 = abcSMC(abcinput, 1000, 1/2, 50000, adaptive=true, store_init=true);

##Look at weights
smcoutput1.abcdists[1].w
[smcoutput2.abcdists[i].w for i in 1:smcoutput2.niterations]

##Plot simulations from each importance density and acceptance regions
nits = min(smcoutput1.niterations, smcoutput2.niterations)
PyPlot.figure(figsize=(12,12))
PyPlot.subplot(221)
for i in 1:nits
    plot_init(smcoutput1, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$")
PyPlot.title("Non-adaptive simulations")
PyPlot.subplot(222)
for i in 1:nits
    plot_init(smcoutput2, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$") 
PyPlot.title("Adaptive simulations")
PyPlot.subplot(223)
for i in 1:nits
    plot_acc(smcoutput1, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$")
PyPlot.title("Non-adaptive acceptance regions")
PyPlot.subplot(224)
for i in 1:nits
    plot_acc(smcoutput2, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$")
PyPlot.title("Adaptive acceptance regions")
PyPlot.tight_layout()
PyPlot.savefig("normal_acc_regions.pdf")

##Plot RMSEs
PyPlot.figure(figsize=(9,3))
RMSE1 = [ sqrt(sum(smcoutput1.parameters[1,:,i].^2)) for i in 1:nits ]
RMSE2 = [ sqrt(sum(smcoutput2.parameters[1,:,i].^2)) for i in 1:nits ]   
plot(smcoutput1.cusims[1:nits], RMSE1, "r-o")
plot(smcoutput2.cusims[1:nits], RMSE2, "b-o")
xlabel("Simulations")
ylabel("RMSE")
PyPlot.tight_layout()
PyPlot.savefig("normal_RMSE.pdf")

######################################################################
##EXAMPLE 2: 2nd sum stat has non-constant variance
######################################################################
function sample_sumstats(pars::Array)
    success = true
    theta = pars[1]
    stats = [theta + 0.1*randn(1), (1.0+10.0/(1.0+(theta/50.0)^2))*randn(1)]
    (success, stats)
end

sobs = [0.0,0.0]
    
abcinput = ABCInput();
abcinput.prior = MvNormal(1, 100.0);
abcinput.sample_sumstats = sample_sumstats;
abcinput.abcdist = WeightedEuclidean(sobs);
abcinput.sobs = sobs;
abcinput.nsumstats = 2;

##Perform ABC-SMC
srand(20)
smcoutput3 = abcSMC(abcinput, 1000, 1/2, 50000, store_init=true);
smcoutput4 = abcSMC(abcinput, 1000, 1/2, 50000, adaptive=true, store_init=true);

nits = min(smcoutput3.niterations, smcoutput4.niterations)
PyPlot.figure()
PyPlot.subplot(221)
for i in 1:nits
    plot_init(smcoutput3, i)
end
PyPlot.subplot(222)
for i in 1:nits
    plot_init(smcoutput4, i)
end 
PyPlot.subplot(223)
for i in 1:nits
    plot_acc(smcoutput3, i)
end
PyPlot.subplot(224)
for i in 1:nits
    plot_acc(smcoutput4, i)
end

######################################################################
##EXAMPLE 3: 2nd sum stat matches model poorly
######################################################################
function sample_sumstats(pars::Array)
    success = true
    theta = pars[1]
    stats = [theta + 0.1*randn(1), randn(1)]
    (success, stats)
end

sobs = [0.0,3.0];
    
abcinput = ABCInput();
abcinput.prior = MvNormal(1, 100.0);
abcinput.sample_sumstats = sample_sumstats;
abcinput.abcdist = WeightedEuclidean(sobs);
abcinput.sobs = sobs;
abcinput.nsumstats = 2;

##Perform ABC-SMC
srand(20);
smcoutput5 = abcSMC(abcinput, 1000, 1/2, 250000, store_init=true);
srand(20);
smcoutput6 = abcSMC(abcinput, 1000, 1/2, 250000, adaptive=true, store_init=true);
abcinput.abcdist = WeightedEuclidean(sobs, "ADO");
srand(20);
smcoutput7 = abcSMC(abcinput, 1000, 1/2, 250000, adaptive=true, store_init=true);

nits = min(smcoutput5.niterations, smcoutput6.niterations, smcoutput7.niterations)
PyPlot.figure(figsize=(12,12))
PyPlot.subplot(131)
for i in 1:nits
    plot_init(smcoutput5, i)
end
PyPlot.subplot(132)
for i in 1:nits
    plot_init(smcoutput6, i)
end
PyPlot.subplot(133)
for i in 1:nits
    plot_init(smcoutput7, i)
end    
PyPlot.subplot(131)
for i in 1:nits
    plot_acc(smcoutput5, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$")
PyPlot.title("Non-adaptive")
PyPlot.subplot(132)
for i in 1:nits
    plot_acc(smcoutput6, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$")
PyPlot.title("Adaptive MAD")
PyPlot.subplot(133)
for i in 1:nits
    plot_acc(smcoutput7, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$")
PyPlot.title("Adaptive MADO")
PyPlot.tight_layout()
PyPlot.savefig("normal_acc_regions2.pdf")
   
PyPlot.figure(figsize=(9,3))
RMSE5 = [ sqrt(sum(smcoutput5.parameters[1,:,i].^2)) for i in 1:nits ]
RMSE6 = [ sqrt(sum(smcoutput6.parameters[1,:,i].^2)) for i in 1:nits ]
RMSE7 = [ sqrt(sum(smcoutput7.parameters[1,:,i].^2)) for i in 1:nits ]   
plot(smcoutput5.cusims[1:nits], RMSE5, "r-o")
plot(smcoutput6.cusims[1:nits], RMSE6, "b-o")
plot(smcoutput7.cusims[1:nits], RMSE7, "g-o")
xlabel("Simulations")
ylabel("RMSE")  
