using ABCDistances
using Distributions
using PyPlot

######################################################################
##Define plotting functions
######################################################################
plot_cols = ("b", "g", "r", "c", "m", "y", "k");
function plot_init(out::ABCPMCOutput, i::Int32)
    ssim = out.init_sims[i]
    n = min(2500, size(ssim)[2])
    s1 = vec(ssim[1,1:n])
    s2 = vec(ssim[2,1:n])
    plot(s1, s2, ".", color=plot_cols[i])
    return nothing
end
function plot_acc(out::ABCPMCOutput, i::Int32)
    w = out.abcdists[i].w
    h = out.thresholds[i]
    h < Inf || return nothing
    ##Plot appropriate ellipse
    θ = 0:0.1:6.3
    x = (h/w[1])*sin(θ)+sobs[1]
    y = (h/w[2])*cos(θ)+sobs[2]
    plot(x, y, lw=3, color=plot_cols[i])
    return nothing
end

######################################################################
##Run ABC
######################################################################
##Set up abcinput
function sample_sumstats(pars::Array)
    success = true
    stats = [pars[1] + 0.1*randn(1); randn(1)]
    (success, stats)
end

sobs = [0.0,0.0];

abcinput = ABCInput();
abcinput.prior = MvNormal(1, 100.0);
abcinput.sample_sumstats = sample_sumstats;
abcinput.abcdist = WeightedEuclidean(sobs);
abcinput.nsumstats = 2;

##Perform ABC-PMC
srand(20);
pmcoutput_adapt = abcPMC(abcinput, 1000, 1/2, 50000, adaptive=true, store_init=true);
srand(20);
abcinput.abcdist = pmcoutput_adapt.abcdists[1];
pmcoutput_adapt_dev = abcPMC_dev(abcinput, 1000, 1/2, 50000, h1=pmcoutput_adapt.thresholds[1], store_init=true);
srand(20);
pmcoutput_noadapt = abcPMC_comparison(abcinput, 1000, 1/2, 50000, h1=pmcoutput_adapt.thresholds[1], initialise_dist=false, store_init=true);
srand(20);
abcinput.abcdist = WeightedEuclidean(sobs);
pmcoutput_noadapt2 = abcPMC(abcinput, 1000, 1/2, 50000, adaptive=false, store_init=true);
srand(20);
abcinput.abcdist = MahalanobisEmp(sobs);
pmcoutput_Mahalanobis = abcPMC(abcinput, 1000, 1/2, 50000, adaptive=true, store_init=true);

##Look at weights
pmcoutput_noadapt.abcdists[1].w
pmcoutput_noadapt2.abcdists[1].w ##The same
[pmcoutput_adapt.abcdists[i].w for i in 1:pmcoutput_adapt.niterations]
[pmcoutput_adapt_dev.abcdists[i].w for i in 1:pmcoutput_adapt_dev.niterations]

##Plot simulations from each importance density and acceptance regions
##For the two analyses considered in talk
nits = min(pmcoutput_noadapt.niterations, pmcoutput_adapt.niterations)
PyPlot.figure(figsize=(12,12))
PyPlot.subplot(221)
for i in 1:nits
    plot_init(pmcoutput_noadapt, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$")
PyPlot.title("Simulations: non-adaptive algorithm")
PyPlot.subplot(222)
for i in 1:nits
    plot_init(pmcoutput_adapt_dev, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$") 
PyPlot.title("Simulations: adaptive algorithm")
PyPlot.subplot(223)
for i in 1:nits
    plot_acc(pmcoutput_noadapt, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$")
PyPlot.title("Acceptance regions: non-adaptive algorithm")
PyPlot.subplot(224)
for i in 1:nits
    plot_acc(pmcoutput_adapt_dev, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$")
PyPlot.title("Acceptance regions: adaptive algorithm")
PyPlot.tight_layout()
PyPlot.savefig("normal_acc_regions.pdf")

##Plot MSEs
get_mses(p, w) = sum(p.^2.*w) / sum(w)
s1 = pmcoutput_noadapt;
s2 = pmcoutput_adapt_dev;
MSE1 = Float64[get_mses(s1.parameters[1,:,i], s1.weights[:,i])  for i in 1:nits];
MSE2 = Float64[get_mses(s2.parameters[1,:,i], s2.weights[:,i])  for i in 1:nits];
PyPlot.figure(figsize=(12,3));
plot(pmcoutput_noadapt.cusims[1:nits], log10(MSE1), "b-o");
plot(pmcoutput_adapt.cusims[1:nits], log10(MSE2), "g-^");
xlabel("Simulations");
ylabel("log₁₀(MSE)");
legend(["Non-adaptive","Adaptive"]);
PyPlot.tight_layout();
PyPlot.savefig("normal_MSE.pdf");
