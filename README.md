# ABCDistances

This pacakge contains several Approximate Bayesian Computation (ABC) algorithms.

It is created for the forthcoming paper "An adaptive ABC distance function",
with the aim of comparing various approaches to distance selection.
Code in the `examples` directory performs the analyses in the paper.

This document gives a quick example of use then documents the commands.

**Warning:** This package is still a work in progress

## Example

First code the model and summary statistics of interest must be set up. Here the model is the g-and-k distribution and the summary statistics are order statistics. Some code for this distribution is included in the package.

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

Now an ABC algorithm can be run. The following commands run an ABC rejection algorithm.

The first command (3rd argument is an integer) returns the 200 best fitting simulations from 10000 total.

The second command (3rd argument is floating point) returns any simulations from 10000 total with distance below the threshold specified.

```julia
abcRejection(abcinput, 10000, 200)
abcRejection(abcinput, 10000, 0.3)
```

The next command runs an ABC SMC algorithm.
See documentation for details of the arguments.

```julia
out = abcSMC(abcinput, 200, 1/2, 10000);
```

Marginal estimates of parameter mean and variances can be calculated from ABC SMC output as follows.
Note that in this example the last 2 parameters are more accurately estimated by the adaptive method.

```julia
parameter_means(out)
parameter_vars(out)
```

##Documentation

###Performing ABC

####`ABCInput`

This type contains input needed for any of the ABC algorithms.
Its fields are:

* `prior`
A `DiscreteMultivariateDistribution` or `ContinuousMultivariateDistribution` variable defining the prior distribution. These are defined in the `Distributions` package.
* `sample_sumstats`
A function taking a (Float64) vector of parameters as input and returning `(success, stats)`: a boolean and a (Float64) vector of summary statistics. `success=false` indicates a decision that the simulation should be rejected before it was completed. One use case is when simulations from some parameter values are extremely slow.
* `abc_dist`
An object of type `ABCDistance` defining how distance behaves.
* `nsumstats`
Dimension of a summary statistic vector.

There is a convenience constructor `ABCInput()`.

####`abcRejection`

This has various methods.
Possible arguments are as follows:

* `in`
An `ABCInput` variable.

* `nsims`
How many simulations to perform.

* `k`
How many simulations to accept (an integer).

* `h`
Acceptance threshold (floating point).

* `store_init`.
Whether to return simulations used to initialise the distance function.

The methods are:

* `(in, nsims, k; store_init)`
Performs ABC accepting k simulations

* `(in, nsims, h; store_init)`
Performs ABC accepting simulations with distance below the threshold h.

* `(in, nsims; store_init)`
Performs ABC accepting everything.

The output is a ABCRejOutput object.

####`abcSMC`

n.b. Shows progress

####`abcSMC_comparison`

###ABC output

####`ABCRejOutput`

n.b. Usually sorted

####`ABCSMCOutput`

####`parameter_means`
####`parameter_vars`
####`parameter_covs`

###Distance functions

###Other code

Code is also provided to perform simulations for the g-and-k and Lotka-Volterra examples.
This is not documented for now.