# ABCNorms

Approximate Bayesian computation with various distance norms.

Work in progress!

## Example

The first code set up the particular model and summary statistics of interest. Here the model is the g-and-k distribution and the summary statistics are particular order statistics. Some code for this distribution is included in the package.

```julia
using ABCDistances
quantiles = [1000,2000,3000,4000,5000,6000,7000,8000,9000]
ndataset = 10000
##Simulate from the model and return summary statistics
function sample_sumstats(pars::Array{Float64,1})
    success = true
    stats = rgk_os(pars, quantiles, ndataset)
    (success, stats)
end

##Generate the observed summary statistics
theta0 = [3.0,1.0,1.5,0.5]
srand(1)
(success, sobs) = sample_sumstats(theta0)
```

Next the prior distribution is specified. This is simply 4 independent Uniform(0,10) random variables.
```julia
using Distributions
import Distributions.length, Distributions._rand!, Distributions._pdf ##So that these can be extended

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
```

Next an `ABCInput` type is created and populated using the above.

Note `abcdist` must be a subtype of `ABCDistance`. Several options are defined in `distances.jl`.

```julia
abcinput = ABCInput();
abcinput.prior = GKPrior();
abcinput.sample_sumstats = sample_sumstats;
abcinput.abcdist = MahalanobisDiag(sobs);
abcinput.sobs = sobs;
abcinput.nsumstats = 9;
```

Now an ABC algorithm can be run. The following commands run an ABC rejection algorithm. The 2nd argument is the number of iterations to perform.

The first command (3rd argument is an integer) returns the 200 best fitting simulations.

The second command (3rd argument is floating point) returns all simulations with distance below the threshold specified.

```julia
abcRejection(abcinput, 10000, 200)
abcRejection(abcinput, 10000, 0.3)
```

The next command runs an ABC SMC algorithm.
The second argument is how many simulations to produce at each iteration which are below the *previous* threshold.
The third is how simulations to accept at each iteration.
The last argument is how many simulations to perform in total - the algorithm terminates once this is reached.

```julia
out = abcSMC(abcinput, 2000, 200, 100000);
```

The following command instead uses an adaptive distance function (an ongoing research project).

```julia
out_adapt = abcSMC(abcinput, 2000, 200, 100000; adaptive=true);
```

Marginal estimates of parameter mean and variances can be calculated from ABC SMC output as follows.
Note that in this example the last 2 parameters are more accurately estimated by the adaptive method.

```julia
parameter_means(out)
parameter_means(out_adapt)
parameter_vars(out)
parameter_vars(out_adapt)
```

