##Test ABC algorithms and related functions.
##This is done by running a simple normal example.
##It is simply tested that these do not produce errors.
using ABCDistances
using Distributions

##Set up abcinput
function sample_sumstats(pars::Array)
    success = true
    stats = [pars[1] + 0.1*randn(1); randn(1)]
    (success, stats)
end

sobs = [0.0,0.0]

abcinput = ABCInput();
abcinput.prior = MvNormal(1, 100.0);
abcinput.sample_sumstats = sample_sumstats;
abcinput.abcdist = WeightedEuclidean(sobs);
abcinput.nsumstats = 2;

##ABC rejection
srand(1)
out1 = abcRejection(abcinput, 1000, 200)
abcRejection(abcinput, 1000, 0.1)

##ABC-PMC (doesn't quite test all options)
srand(1)
out2 = abcPMC3V(abcinput, 200, 1/2, 5000, silent=true)
abcPMC5(abcinput, 200, 1/2, 5000, silent=true)
abcPMC3V(abcinput, 200, 1/2, 5000, store_init=true, silent=true)
abcPMC3V(abcinput, 200, 1/2, 5000, store_init=true, silent=true)

abcPMC3(abcinput, 200, 1/2, 5000, silent=true)
abcPMC3(abcinput, 200, 1/2, 5000, store_init=true, silent=true)
abcPMC3(abcinput, 200, 1/2, 5000, diag_perturb=true, silent=true)

abcPMC4(abcinput, 200, 1/2, 5000, silent=true);
abcPMC4(abcinput, 200, 1/2, 5000, store_init=true, silent=true);

##Mean and variance functions
parameter_means(out1)
parameter_means(out2)
parameter_vars(out1)
parameter_vars(out2)
