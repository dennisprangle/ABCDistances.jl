using ABCDistances
using Distributions
Libdl.dlopen("/usr/lib/liblapack.so.3", Libdl.RTLD_GLOBAL); ##Needed to avoid PyPlot problems on my work machine
##I sometimes also need run "export LD_LIBRARY_PATH=$HOME/.julia/v0.6/Conda/deps/usr/lib/:$LD_LIBRARY_PATH" - see https://github.com/JuliaPy/Conda.jl/issues/105
using PyPlot

######################################################################
##Define plotting functions
######################################################################
plot_cols = ("b", "g", "r", "c", "m", "y", "k");
function plot_init(out::ABCPMCOutput, i::Int)
    ssim = out.init_sims[i]
    n = min(2500, size(ssim)[2])
    s1 = vec(ssim[1,1:n])
    s2 = vec(ssim[2,1:n])
    plot(s1, s2, ".", color=plot_cols[i])
    return nothing
end
function plot_acc(out::ABCPMCOutput, i::Int)
    w = out.abcdists[i].w
    h = out.thresholds[i]
    h < Inf || return nothing
    ##Plot appropriate ellipse
    θ = 0:0.1:6.3
    x = (h/w[1])*sin.(θ)+sobs[1]
    y = (h/w[2])*cos.(θ)+sobs[2]
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
pmcoutput_alg5 = abcPMC5(abcinput, 1000, 1/2, 50000, store_init=true);
srand(20);
abcinput.abcdist = pmcoutput_alg5.abcdists[1];
pmcoutput_alg4 = abcPMC4(abcinput, 1000, 1/2, 50000, h1=pmcoutput_alg5.thresholds[1], store_init=true);
srand(20);
pmcoutput_alg2 = abcPMC2(abcinput, 1000, 1/2, 50000, h1=pmcoutput_alg5.thresholds[1], store_init=true);
srand(20);
abcinput.abcdist = WeightedEuclidean(sobs);
pmcoutput_alg3V = abcPMC3V(abcinput, 1000, 1/2, 50000, store_init=true);
srand(20);
abcinput.abcdist = MahalanobisEmp(sobs);
pmcoutput_alg5_Mahalanobis = abcPMC5(abcinput, 1000, 1/2, 50000, store_init=true);

##Look at weights
pmcoutput_alg2.abcdists[1].w
pmcoutput_alg3V.abcdists[1].w ##The same
[pmcoutput_alg4.abcdists[i].w for i in 1:pmcoutput_alg4.niterations]
[pmcoutput_alg5.abcdists[i].w for i in 1:pmcoutput_alg5.niterations]

##Plot simulations from each importance density and acceptance regions
##First for the three analyses shown in the paper
nits = min(pmcoutput_alg2.niterations, pmcoutput_alg5.niterations, pmcoutput_alg4.niterations)
PyPlot.figure(figsize=(13,12))
PyPlot.subplot(231)
for i in 1:nits
    plot_init(pmcoutput_alg2, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$")
PyPlot.title("Algorithm 2 simulations")
PyPlot.subplot(232)
for i in 1:nits
    plot_init(pmcoutput_alg4, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$") 
PyPlot.title("Algorithm 4 simulations")
PyPlot.subplot(233)
for i in 1:nits
    plot_init(pmcoutput_alg5, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$")
PyPlot.title("Algorithm 5 simulations")
PyPlot.subplot(234)
for i in 1:nits
    plot_acc(pmcoutput_alg2, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$")
PyPlot.title("Algorithm 2 acceptance regions")
PyPlot.subplot(235)
for i in 1:nits
    plot_acc(pmcoutput_alg4, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$")
PyPlot.title("Algorithm 4 acceptance regions")
PyPlot.subplot(236)
for i in 1:nits
    plot_acc(pmcoutput_alg5, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$")
PyPlot.title("Algorithm 5 acceptance regions")
PyPlot.tight_layout()
PyPlot.savefig("normal_acc_regions.pdf")

##Preceding plot for two analyses omitted from the paper
##(Mahalanobis acc region omitted as this would need extra code)
nits = min(pmcoutput_alg3V.niterations, pmcoutput_alg5_Mahalanobis.niterations)
PyPlot.figure(figsize=(12,12))
PyPlot.subplot(221)
for i in 1:nits
    plot_init(pmcoutput_alg3V, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$")
PyPlot.subplot(222)
for i in 1:nits
    plot_init(pmcoutput_alg5_Mahalanobis, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$") 
PyPlot.subplot(223)
for i in 1:nits
    plot_acc(pmcoutput_alg3V, i)
end
PyPlot.xlabel(L"$s_1$")
PyPlot.ylabel(L"$s_2$")
PyPlot.tight_layout()

##Plot MSEs
get_mses(p, w) = sum(p.^2.*w) / sum(w)
s1 = pmcoutput_alg2;
s2 = pmcoutput_alg4;
s3 = pmcoutput_alg5;
MSE1 = Float64[get_mses(s1.parameters[1,:,i], s1.weights[:,i])  for i in 1:s1.niterations];
MSE2 = Float64[get_mses(s2.parameters[1,:,i], s2.weights[:,i])  for i in 1:s2.niterations];
MSE3 = Float64[get_mses(s3.parameters[1,:,i], s3.weights[:,i])  for i in 1:s3.niterations];
PyPlot.figure(figsize=(12,3));
plot(s1.cusims, log10.(MSE1), "b-o");
plot(s2.cusims, log10.(MSE2), "g-^");
plot(s3.cusims, log10.(MSE3), "y-*");
xlabel("Simulations");
ylabel("log₁₀(MSE)");
legend(["Algorithm 2","Algorithm 4","Algorithm 5"], loc="lower left");
PyPlot.tight_layout();
PyPlot.savefig("normal_MSE.pdf");

##Add lines for other algorithms
s4 = pmcoutput_alg3V;
s5 = pmcoutput_alg5_Mahalanobis;
MSE4 = Float64[get_mses(s4.parameters[1,:,i], s4.weights[:,i])  for i in 1:nits];
MSE5 = Float64[get_mses(s5.parameters[1,:,i], s5.weights[:,i])  for i in 1:nits];
plot(s4.cusims[1:nits], log10.(MSE4), "r-x");
plot(s5.cusims[1:nits], log10.(MSE5), "k-|");
legend(["Algorithm 2", "Algorithm 4", "Alg 5", "Alg 3 (variant)", "Alg 5 Mahalanobis"]);
    
#####################################################    
##Exploratory investigation of different alpha values
##(as requested by reviewers)
#####################################################
alphas = 0.05:0.05:0.95;
MSEs_alg5 = zeros(alphas);
MSEs_alg4 = zeros(alphas);
MSEs_alg3 = zeros(alphas);
for i in 1:length(alphas)
    abcinput.abcdist = WeightedEuclidean(sobs)
    srand(20)
    x = abcPMC5(abcinput, 1000, alphas[i], 50000)
    nits = x.niterations
    MSEs_alg5[i] = (nits == 0) ? Inf : get_mses(x.parameters[1,:,nits], x.weights[:,nits])
    srand(20)
    x = abcPMC3(abcinput, 1000, alphas[i], 50000)
    nits = x.niterations
    MSEs_alg3[i] = (nits == 0) ? Inf : get_mses(x.parameters[1,:,nits], x.weights[:,nits])
    if nits > 0
        abcinput.abcdist = x.abcdists[1]
    end
    srand(20)
    x = abcPMC4(abcinput, 1000, alphas[i], 50000)
    nits = x.niterations
    MSEs_alg4[i] = (nits == 0) ? Inf : get_mses(x.parameters[1,:,nits], x.weights[:,nits])    
end


PyPlot.figure();
plot(alphas, log10.(MSEs_alg5), "r-x");
plot(alphas, log10.(MSEs_alg3), "b-o");
plot(alphas, log10.(MSEs_alg4), "g-*");
##MSE is minimised by alpha approximately equal to 0.5
