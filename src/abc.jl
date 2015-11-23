#####################################
##General ABC code
##i.e. type definitions and utilities
#####################################

##################
##TYPE DEFINITIONS
#################
"
This type contains input needed for any of the ABC algorithms.
Its fields are:

* `prior`
A `DiscreteMultivariateDistribution` or `ContinuousMultivariateDistribution` variable defining the prior distribution. These are defined in the `Distributions` package.
* `sample_sumstats`
A function taking a (`Float64`) vector of parameters as input and returning `(success, stats)`: a boolean and a (`Float64`) vector of summary statistics. `success=false` indicates a decision that the simulation should be rejected before it was completed. One use case is when simulations from some parameter values are extremely slow.
* `abc_dist`
A variable of type `ABCDistance` defining how distance behaves.
* `nsumstats`
Dimension of a summary statistic vector.

There is a convenience constructor `ABCInput()`.
"
type ABCInput
    prior::Union{DiscreteMultivariateDistribution, ContinuousMultivariateDistribution}
    sample_sumstats::Function
    abcdist::ABCDistance
    nsumstats::Int
end

##Full results of an ABC analysis
abstract ABCOutput

"
This type contains output from a ABC-rejection algorithm.
Its fields are:

* `nparameters`
Number of parameters.
* `nsumstats`
Number of summary statistics.
* `nsims`
Number of simulations performed.
* `nsuccesses`
Number of successful simulations (i.e. excluding those with success=false).
* `parameters`
`parameters[i,j]` is the ith parameter for the jth accepted simulation.
* `sumstats`
`sumstats[i,j]` is the ith summary statistic for the jth accepted simulation.
* `distances`
`distances[i]` is the distance for the ith accepted simulation.
* `weights`
`weights[i]` is the weight for the ith accepted simulation.
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
"
type ABCRejOutput <: ABCOutput
    nparameters::Int
    nsumstats::Int
    nsims::Int                  ##Number of simulations performed
    nsuccesses::Int             ##Number of successful simulations. Usually equal to nsims.
    parameters::Array{Float64, 2} ##parameters[i,j] is ith parameter for jth accepted sim
    sumstats::Array{Float64, 2}   ##sumstats[i,j] is ith sumstat for jth accepted sim
    distances::Array{Float64, 1}  ##distance[i] is distance for ith accepted sim
    weights::Array{Float64, 1}    ##weights[i] is weight for ith accepted sim
    abcdist::ABCDistance
    init_sims::Array{Float64, 2}  ##sims used for distance initialisation (only stored optionally)
    init_pars::Array{Float64, 2}  ##pars used for distance initialisation (only stored optionally)
end

"
This type contains output from a ABC-PMC algorithm.
Its fields are:

* `nparameters`
Number of parameters.
* `nsumstats`
Number of summary statistics.
* `niterations`
Number of iterations performed (i.e. number of target distributions used).
* `nsims`
Total number of simulations performed.
* `cusims`
`cusims[i]` is the cumulative number of simulations performed up to the end of iteration i.
* `parameters`
`parameters[i,j,k]` is the ith parameter for the jth accepted simulation in iteration k.
* `sumstats`
`sumstats[i,j,k]` is the ith summary statistic for the jth accepted sim in iteration k.
* `distances`
`distances[i,j]` is the distance for ith accepted simulation in iteration j.
* `weights`
`weights[i,j]` is the weight for ith accepted sim in iteration j.
* `abcdists`
`abcdists[i]` the `ABCDistance` type variable defining how distance behaves in iteration i.
* `thresholds`
`thresholds[i] is the acceptance threshold used in iteration i.
* `init_sims`
init_sims[i][j,k] is the jth summary statistic in the kth simulation used in distance initialisation in iteration i. This is only recorded if `store_init` is true. (n.b. This is an array of arrays as the number of simulations used for distance initialisation can vary.)
* `init_pars`
init_pars[i][j,k] is the jth parameter in the kth simulation used in distance initialisation in iteration i. This is only recorded if `store_init` is true. (n.b. This is an array of arrays as the number of simulations used for distance initialisation can vary.)
"
type ABCPMCOutput <: ABCOutput
    nparameters::Int
    nsumstats::Int
    niterations::Int            ##Number of iteration performed
    nsims::Int                  ##Total number of simulations performed
    cusims::Array{Int, 1}       ##cusims[i] is cumulative sims used up to end of iteration i
    parameters::Array{Float64, 3} ##parameters[i,j,k] is ith parameter for jth accepted sim in iteration k
    sumstats::Array{Float64, 3}   ##sumstats[i,j,k] is ith sumstat for jth accepted sim in iteration k
    distances::Array{Float64, 2}  ##distances[i,j] is distance for ith accepted sim in iteration j
    weights::Array{Float64, 2}    ##weights[i,j] is weight for ith accepted sim in iteration j
    abcdists::Array{ABCDistance, 1}    ##abcdist[i] is distance used in iteration i
    thresholds::Array{Float64, 1}  ##threshold[i] is threshold used in iteration i
    init_sims::Array{Array{Float64, 2}, 1} ##init_sims[i] is sims for distance initialisation at iteration i (only stored optionally)
    init_pars::Array{Array{Float64, 2}, 1} ##init_pars[i] is pars for distance initialisation at iteration i (only stored optionally)
end

##TO DO: ABCPMCOutput needs a show function

#################
##CONSTRUCTORS
#################
##Semi-sensible defaults
function ABCInput()
    ABCInput(MvNormal(1, 1.0),  ##prior
             (x)->rand(1),       ##sample_sumstats draws from U(0,1) independent of parameters
             Euclidean([1.0]),  ##abcdist
             1)                  ##nsumstats
end

#################
##UTILITY METHODS
#################
function show(io::IO, out::ABCRejOutput)
    (p,k) = size(out.parameters)
    means = parameter_means(out)
    CI_lower = Array(Float64, p)
    CI_upper = Array(Float64, p)
    for i in 1:p
        y = squeeze(out.parameters[i,:], 1)
        (CI_lower[i], CI_upper[i]) = quantile(y, WeightVec(out.weights), [0.025,0.975])
    end
    print("ABC output, $k accepted values from $(out.nsims) simulations\n")
    ess = sum(out.weights)^2 / sum(out.weights.^2)
    print("Effective sample size $(round(ess,1))\n")
    print("Means and 95% credible intervals:\n")
    for (i in 1:p)
        @printf("Parameter %d: %.2e (%.2e,%.2e)\n", i, means[i], CI_lower[i], CI_upper[i])
    end
end

function copy(out::ABCRejOutput)
    ABCRejOutput(out.nparameters, out.nsumstats, out.nsims, out.nsuccesses, out.parameters, out.sumstats, out.distances, out.weights, out.abcdist, out.init_sims, out.init_pars)
end

##Sort output into distance order
function sortABCOutput!(out::ABCRejOutput)
    ##Sort results into closeness order
    closenessorder = sortperm(out.distances) ##nb This uses the default algorithm mergesort which leaves ties in the original order
    out.parameters = out.parameters[:,closenessorder]
    out.sumstats = out.sumstats[:,closenessorder]
    out.distances = out.distances[closenessorder]
    out.weights = out.weights[closenessorder]
    return
end

"
This reports the estimated posterior means of `out`, a `ABCOutput` object.
If `out` is a `ABCRejOutput` object the output is a vector of parameter means.
If `out` is a `ABCPMCOutput` object the output is a matrix whose columns are parameter means in each iteration.
"
function parameter_means(out::ABCRejOutput)
    vec(mean(out.parameters, WeightVec(out.weights), 2))
end

function parameter_means(out::ABCPMCOutput)
    means = Array(Float64, (out.nparameters, out.niterations))
    for (it in 1:out.niterations)
      means[:, it] = mean(out.parameters[:,:,it], WeightVec(out.weights[:,it]), 2)
    end
    means
end

"
This reports the estimated posterior marginal variances of `out`, a `ABCOutput` object.
If `out` is a `ABCRejOutput` object the output is a vector of parameter variances.
If `out` is a `ABCPMCOutput` object the output is a matrix whose columns are parameter variances in each iteration.
"
function parameter_vars(out::ABCRejOutput)
  vec(var(out.parameters, WeightVec(out.weights), 2))
end

function parameter_vars(out::ABCPMCOutput)
    vars = Array(Float64, (out.nparameters, out.niterations))
    for (it in 1:out.niterations)
      vars[:, it] = var(out.parameters[:,:,it], WeightVec(out.weights[:,it]), 2)
    end
    vars
end

"
This reports the estimated posterior covariance matrix of `out`, a `ABCOutput` object.
If `out` is a `ABCRejOutput` object the output is a covariance matrix.
If `out` is a `ABCPMCOutput` object the output is an array `x` where `x[:,:,i]` is a covariance matrix for the ith iteration.
"
function parameter_covs(out::ABCRejOutput)
    cov(out.parameters, WeightVec(out.weights), vardim=2)
end

function parameter_covs(out::ABCPMCOutput)
    covs = Array(Float64, (out.nparameters, out.nparameters, out.niterations))
    for (it in 1:out.niterations)
      covs[:, :, it] = cov(out.parameters[:,:,it], WeightVec(out.weights[:,it]), vardim=2)
    end
    covs
end
