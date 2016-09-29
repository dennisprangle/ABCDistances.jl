# ABCDistances

This pacakge contains several [Approximate Bayesian Computation](https://en.wikipedia.org/wiki/Approximate_Bayesian_computation) (ABC) algorithms.

It is created for the paper ["Adapting the ABC distance function"](http://arxiv.org/abs/1507.00874)
with the aim of comparing various approaches to distance selection.
Code in the `examples` directory performs the analyses in the paper.

Version 0.0.1 matches the v1 and v2 arXiv submissions and is written for Julia v0.3.
Version 0.1.0 matches the v3 arXiv submission and is written for Julia v0.4.
Version 0.1.1 updates v0.1.0 for Julia v0.5.

This document gives a quick usage example then documents the commands.

## Example

First code for the model and summary statistics of interest must be set up. Here the model is the [g-and-k distribution](http://link.springer.com/article/10.1023%2FA%3A1013120305780?LI=true) and the summary statistics are order statistics. Code for simulating from this distribution is included in the package.

```julia
using ABCDistances;
quantiles = [1250*i for i in 1:7];
ndataset = 10000;
##Simulate from the model and return summary statistics
function sample_sumstats(pars::Array{Float64,1})
    success = true
    stats = rgk_os(pars, quantiles, ndataset)
    (success, stats)
end

##Generate the observed summary statistics
theta0 = [3.0,1.0,1.5,0.5];
srand(1);
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
abcinput.abcdist = WeightedEuclidean(sobs);
abcinput.nsumstats = 7;
```

Now an ABC algorithm can be run. The following commands run an ABC-rejection algorithm.

The first command (3rd argument is an integer) returns the 200 best fitting simulations from 10000 total.

The second command (3rd argument is floating point) returns any simulations from 10000 total with distance below the threshold specified.

```julia
abcRejection(abcinput, 10000, 200)
abcRejection(abcinput, 10000, 0.3)
```

The next command runs an ABC PMC algorithm.
See documentation for details of the arguments.

```julia
out = abcPMC3(abcinput, 200, 1/2, 10000);
```

Marginal estimates of parameter mean and variances can be calculated from ABC PMC output as follows.

```julia
parameter_means(out)
parameter_vars(out)
```