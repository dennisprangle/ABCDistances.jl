################################################
##A trivial example to demonstrate iterative ABC
################################################
using ABCDistances
using Distributions

######################################################################
##Run ABC
######################################################################
##Set up abcinput
function sample_sumstats(pars::Vector{Float64})
    success = true
    stats = collect(pars[1] + randn(1))
    (success, stats)
end

sobs = [30.0];

abcinput = ABCInput();
abcinput.prior = MvNormal(1, 10.0);
abcinput.sample_sumstats = sample_sumstats;
abcinput.abcdist = Euclidean(sobs);
abcinput.nsumstats = 1;

##Perform ABC-PMC
srand(20);
pmcout = abcPMC_comparison(abcinput, 1000, 1/2, 50000, h1=20.0, initialise_dist=false, store_init=true);

##Export output to plot in R
writedlm("sc_pars_acc.txt", squeeze(pmcout.parameters, 1));
writedlm("sc_weights.txt", pmcout.weights);
nits = pmcout.niterations;
init_pars = Array(Float64, (1000, nits));
for i in 1:nits
    init_pars[:,i] = pmcout.init_pars[i][1, 1:1000]
end
writedlm("sc_pars_all.txt", init_pars);
