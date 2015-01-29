# ABCNorms

Approximate Bayesian computation with various distance norms.

Work in progress!

## Example

First some code to set up the particular problem of interest. This is for infering the parameter of the g-and-k distribution. Some code for this distribution is included in this package.

```julia
using ABCDistances

##Sample from the prior
function rprior()
    10.0*rand(4)
end

##Evaluate the prior density
function dprior(pars::Array) ##Independent uniform priors on [0,10]
    if (any(pars.<0.0))
        return 0.0
    elseif (any(pars.>10.0))
        return 0.0
    end
    return 1.0
end

quantiles = [1000,2000,3000,4000,5000,6000,7000,8000,9000]
ndataset = 10000
##Simulate from the model and return summary statistics
function sample_sumstats(pars::Array{Float64,1})
    rgk_os(pars, quantiles, ndataset)
end

##Generate the observed summary statistics
theta0 = [3.0,1.0,1.5,0.5]
sobs = sample_sumstats(theta0)
```

Next an `ABCInput` type is created and populated using the above.

Note `abcnorm` must be a subtype of `ABCNorm`. Several options are defined in `norms.jl`.

```
abcinput = ABCInput()
abcinput.rprior = rprior
abcinput.sample_sumstats = sample_sumstats
abcinput.abcnorm = MahalanobisDiag()
abcinput.sobs = sobs
abcinput.nparameters = 4
abcinput.nsumstats = 9
abcinput.dprior = dprior
```

Now an ABC algorithm can be run. The following commands run an ABC rejection algorithm. The 2nd argument is the number of iterations to perform.

The first command (3rd argument is an integer) returns the 200 best fitting simulations.

The second command (3rd argument is floating point) returns all simulations with distance below the threshold specified.

```
abcRejection(abcinput, 10000, 200)
abcRejection(abcinput, 10000, 0.3)
```

The next command runs an ABC SMC algorithm.
The second argument is how many simulations to accept at each iteration.
The third is how many to use to produce the next threshold.
(So below the next threshold uses the 10% quantile.)
The last argument is how many simulations to perform - the algorithm terminates once this is reached.

```
abcSMC(abcinput, 2000, 200, 1000000);
```