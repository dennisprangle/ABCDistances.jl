# ABCDistances

This pacakge contains several [Approximate Bayesian Computation](https://en.wikipedia.org/wiki/Approximate_Bayesian_computation) (ABC) algorithms.

It is created for the paper ["Adapting the ABC distance function"](http://arxiv.org/abs/1507.00874)
with the aim of comparing various approaches to distance selection.
Code in the `examples` directory performs the analyses in the paper.

This document gives a quick usage example then documents the commands.

**Warning:** This package is still a work in progress

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
out = abcPMC(abcinput, 200, 1/2, 10000);
```

Marginal estimates of parameter mean and variances can be calculated from ABC PMC output as follows.
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
A variable of type `ABCDistance` defining how distance behaves.
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

####`abcPMC`

Perform a version of ABC-PMC.
This is the new algorithm proposed by the paper, Algorithm 4.
Its arguments are as follows:

* `abcinput`
An `ABCInput` variable.

* `N`
Number of accepted particles in each iteration.

* `α`
A tuning parameter between 0 and 1 determining how fast the acceptance threshold is reduced. (In more detai, the acceptance threshold in iteration t is the α quantile of distances from particles which meet the acceptance criteria of the previous iteration.)

* `maxsims`
The algorithm terminates once this many simulations have been performed.

* `nsims_for_init`
How many simulations are stored to initialise the distance function (by default 10,000).

* `adaptive` (optional)
Whether the distance function should be adapted at each iteration.
By default this is false, giving the variant algorithm mentioned in Section 4.1 of the paper.
When true Algorithm 4 of the paper is implemented.

* `store_init` (optional)
Whether to store the parameters and simulations used to initialise the distance function in each iteration (useful to produce many of the paper's figures).
These are stored for every iteration even if `adaptive` is false.
By default false.

* `diag_perturb` (optional)
Whether perturbation should be based on a diagonalised estimate of the covariance.
By default this is false, giving the perturbation described in Section 2.2 of the paper.

* `silent` (optional)
When true no progress bar or messages are shown during execution.
By default false.

The output is a ABCPMCOutput object.

####`abcPMC_comparison`

Perform a version of ABC-PMC.
This is one of the older algorithms reviewed by the paper, Algorithm 2 or 3.
Arguments are as for `abcPMC` with one removal (`adaptive`) and two additions

* `initialise_dist` (optional)
If true (the default) then the distance will be initialised at the end of the first iteration, giving Algorithm 3. If false a distance must be specified which does not need initialisation, giving Algorithm 2.

* `h1` (optional)
The acceptance threshold for the first iteration, by default Inf (accept everything).
If `initialise_dist` is true, then this must be left at its default value.

The output is a ABCPMCOutput object.

###ABC output

####`ABCRejOutput`

This type contains output from a ABC-rejection algorithm.
Its fields are:

* `nparameters`
Number of parameters.
* `nsumstats`
Number of summary statistics
* `nsims`
Number of simulations performed
* `nsuccesses`
Number of successful simulations (i.e. excluding those with success=false)
* `parameters`
`parameters[i,j]` is the ith parameter for the jth accepted simulation
* `sumstats`
`sumstats[i,j]` is the ith summary statistic for the jth accepted simulation
* `distances`
`distances[i]` is the distance for the ith accepted simulation
* `weights`
`weights[i]` is the weight for the ith accepted simulation
* `abcdist`
A variable of type `ABCDistance` defining how distance behaves.
If the distance requires weights etc to be set based on simulations, then these details are stored in this variable.
* `init_sims`
`init_sims[i,j]` is the ith summary statistic in the jth simulation used in distance initialisation. This is only recorded if `store_init` is true.
* `init_pars`
`init_pars[i,j]` is the ith parameter in the jth simulation used in distance initialisation. This is only recorded if `store_init` is true.

Note that the simulations are usually stored in order of increasing distance.
This is guaranteed if this variable is the output of `abcRejection`.

A `show` method exists to give a concise summary of a `ABCRejOutput` variable.

####`ABCPMCOutput`

This type contains output from a ABC-PMC algorithm.
Its fields are:

* `nparameters`
Number of parameters.
* `nsumstats`
Number of summary statistics
* `niterations`
Number of iterations performed (i.e. number of target distributions used)
* `nsims`
Total number of simulations performed
* `cusims`
`cusims[i]` is the cumulative number of simulations performed up to the end of iteration i
* `parameters`
`parameters[i,j,k]` is the ith parameter for the jth accepted simulation in iteration k
* `sumstats`
`sumstats[i,j,k]` is the ith summary statistic for the jth accepted sim in iteration k
* `distances`
`distances[i,j]` is the distance for ith accepted simulation in iteration j
* `weights`
`weights[i,j]` is the weight for ith accepted sim in iteration j
* `abcdists`
`abcdists[i]` the `ABCDistance` type variable defining how distance behaves in iteration i
* `thresholds`
`thresholds[i] is the acceptance threshold used in iteration i
* `init_sims`
##init_sims[i][j,k] is the jth summary statistic in the kth simulation used in distance initialisation in iteration i. This is only recorded if `store_init` is true. (n.b. This is an array of arrays as the number of simulations used for distance initialisation can vary.)
* `init_pars`
##init_pars[i][j,k] is the jth parameter in the kth simulation used in distance initialisation in iteration i. This is only recorded if `store_init` is true. (n.b. This is an array of arrays as the number of simulations used for distance initialisation can vary.)

####`parameter_means`, `parameter_vars`, `parameter_covs`

These functions take a single `ABCRejOutput` or `ABCPMCOutput` argument and return information on the moments of the ABC output.

* `parameter_means(out::ABCRejOutput)`
Output is a vector of parameter means.
* `parameter_means(out:ABCPMCOutput)`
Output is a matrix whose columns are parameter means in each iteration.
* `parameter_vars(out::ABCRejOutput)`
Output is a vector of parameter variances.
* `parameter_vars(out:ABCPMCOutput)`
Output is a matrix whose columns are parameter variances in each iteration.
* `parameter_covs(out::ABCRejOutput)`
Output is a parameter variance matrix.
* `parameter_vars(out:ABCPMCOutput)`
Outputs `x` where `x[:,:,i]` is a parameter variance matrix for ith iteration.

###Distance functions

Several `ABCDistance` subtypes are defined.
All require that the observed summaries `sobs` are specified.
Some require other pieces of information, some of which can be initialised during an ABC algorithm
(by calling the `init` method for the ABCDistance variable).
`abcRejection` always attempts this.
So does `abcPMC` on the first iteration, or all iterations for `adaptive=true`.
`abcPMC_comparison` does this on the first iteration if `initialise_dist=true` or otherwise not at all.

* `Euclidean`
This is Euclidean distance. Its constructor is `Euclidean(sobs)`.
* `WeightedEuclidean`
This is weighted Euclidean distance with summary statistics are weighted by scalars - equation (3) in the paper.
It has several constructors.
  * `WeightedEuclidean(sobs, w, scale_type)`
  Here `w` is the weights for each summary and `scale_type` is how to initialise the scale estimate. Possible values for the latter include `"sd"` and `"MAD"` (standard deviation and median absolute deviation).
  * `WeightedEuclidean(sobs, scale_type)`
  Here the weights are left undefined until initialisation of the distance takes place.
  * `WeightedEuclidean(sobs)`
  Equivalent to `WeightedEuclidean(sons, "MAD")`
* MahalanobisEmp
This is weighted Euclidean distance with summary statistics weighted by a matrix i.e. $d(x,y)=(x-y)^T W (x-y)$. (This can often be viewed as an empirical estimate of Mahalanobis distance.)
It has two constructors.
  * MahalanobisEmp(sobs, Ω)
  Where `Ω` is used as the matrix W.
  * MahalanobisEmp(sobs)
  Here the weights are left undefined until initialisation of the distance takes place.	

###Other code

Code is also provided to perform simulations for the g-and-k and Lotka-Volterra examples.
This is not documented for now.