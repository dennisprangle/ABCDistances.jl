using ABCDistances
using Distributions
Libdl.dlopen("/usr/lib/liblapack.so.3", Libdl.RTLD_GLOBAL); ##Needed to avoid PyPlot problems on my work machine
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
##First for the two analyses shown in the paper
nits = min(pmcoutput_noadapt.niterations, pmcoutput_adapt.niterations)
PyPlot.figure(figsize=(12,12))
PyPlot.subplot(221)
for i in 1:nits
    plot_init(pmcoutput_noadapt, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$")
PyPlot.title("Algorithm 2 simulations")
PyPlot.subplot(222)
for i in 1:nits
    plot_init(pmcoutput_adapt, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$") 
PyPlot.title("Algorithm 4 simulations")
PyPlot.subplot(223)
for i in 1:nits
    plot_acc(pmcoutput_noadapt, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$")
PyPlot.title("Algorithm 2 acceptance regions")
PyPlot.subplot(224)
for i in 1:nits
    plot_acc(pmcoutput_adapt, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$")
PyPlot.title("Algorithm 4 acceptance regions")
PyPlot.tight_layout()
PyPlot.savefig("normal_acc_regions.pdf")

##Preceding plot for three analyses omitted from the paper
##(Mahalanobis acc region omitted as this would need extra code)
nits = min(pmcoutput_noadapt2.niterations, pmcoutput_Mahalanobis.niterations, pmcoutput_adapt_dev.niterations)
PyPlot.figure(figsize=(12,12))
PyPlot.subplot(231)
for i in 1:nits
    plot_init(pmcoutput_noadapt2, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$")
PyPlot.subplot(232)
for i in 1:nits
    plot_init(pmcoutput_Mahalanobis, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$") 
PyPlot.subplot(233)
for i in 1:nits
    plot_acc(pmcoutput_noadapt2, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$")
PyPlot.subplot(234)
for i in 1:nits
    plot_init(pmcoutput_adapt_dev, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$")
PyPlot.title("Algorithm 5 simulations")
PyPlot.subplot(235)
for i in 1:nits
    plot_acc(pmcoutput_adapt_dev, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$")
PyPlot.title("Algorithm 5 acceptance regions")
PyPlot.tight_layout()

##Plot MSEs
get_mses(p, w) = sum(p.^2.*w) / sum(w)
s1 = pmcoutput_noadapt;
s2 = pmcoutput_adapt;
MSE1 = Float64[get_mses(s1.parameters[1,:,i], s1.weights[:,i])  for i in 1:nits];
MSE2 = Float64[get_mses(s2.parameters[1,:,i], s2.weights[:,i])  for i in 1:nits];
PyPlot.figure(figsize=(12,3));
plot(pmcoutput_noadapt.cusims[1:nits], log10(MSE1), "b-o");
plot(pmcoutput_adapt.cusims[1:nits], log10(MSE2), "g-^");
xlabel("Simulations");
ylabel("log₁₀(MSE)");
legend(["Algorithm 2","Algorithm 4"]);
PyPlot.tight_layout();
PyPlot.savefig("normal_MSE.pdf");

##Add lines for other algorithms
s3 = pmcoutput_noadapt2;
s4 = pmcoutput_Mahalanobis;
s5 = pmcoutput_adapt_dev;
MSE3 = Float64[get_mses(s3.parameters[1,:,i], s3.weights[:,i])  for i in 1:nits];
MSE4 = Float64[get_mses(s4.parameters[1,:,i], s4.weights[:,i])  for i in 1:nits];
MSE5 = Float64[get_mses(s5.parameters[1,:,i], s5.weights[:,i])  for i in 1:nits];
plot(pmcoutput_noadapt2.cusims[1:nits], log10(MSE3), "r-x");
plot(pmcoutput_Mahalanobis.cusims[1:nits], log10(MSE4), "k-|");
plot(pmcoutput_Mahalanobis.cusims[1:nits], log10(MSE5), "y-*");
legend(["Algorithm 2","Algorithm 4","Alg 4 (non-adaptive)", "Alg 4 Mahalanobis", "Alg 5"]);

#####################################################    
##Exploratory investigation of different alpha values
##(as requested by reviewers)
#####################################################
alphas = 0.05:0.05:0.95;
MSEs_adaptive = zeros(alphas);
MSEs_adaptive2 = zeros(alphas);
MSEs_nonadaptive = zeros(alphas);
for i in 1:length(alphas)
    abcinput.abcdist = WeightedEuclidean(sobs)
    srand(20)
    x = abcPMC(abcinput, 1000, alphas[i], 50000, adaptive=true)
    nits = x.niterations
    MSEs_adaptive[i] = (nits == 0) ? Inf : get_mses(x.parameters[1,:,nits], x.weights[:,nits])
    srand(20)
    x = abcPMC_comparison(abcinput, 1000, alphas[i], 50000)
    nits = x.niterations
    MSEs_nonadaptive[i] = (nits == 0) ? Inf : get_mses(x.parameters[1,:,nits], x.weights[:,nits])
    if nits > 0
        abcinput.abcdist = x.abcdists[1]
    end
    srand(20)
    x = abcPMC_dev(abcinput, 1000, alphas[i], 50000)
    nits = x.niterations
    MSEs_adaptive2[i] = (nits == 0) ? Inf : get_mses(x.parameters[1,:,nits], x.weights[:,nits])    
end


PyPlot.figure();
plot(alphas, log10(MSEs_adaptive), "r-x");
plot(alphas, log10(MSEs_nonadaptive), "b-o");
plot(alphas, log10(MSEs_adaptive2), "g-*");
##MSE is minimised by alpha approximately equal to 0.5
