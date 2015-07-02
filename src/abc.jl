#####################################
##General ABC code
##i.e. type definitions and utilities
#####################################

##################
##TYPE DEFINITIONS
#################
##Input for an ABC analysis
type ABCInput
    prior::Union(DiscreteMultivariateDistribution, ContinuousMultivariateDistribution)
    sample_sumstats::Function
    abcdist::ABCDistance
    nsumstats::Int32    
end

##Full results of an ABC analysis
abstract ABCOutput

##Rejection sampling output
type ABCRejOutput <: ABCOutput
    nparameters::Int32
    nsumstats::Int32 
    nsims::Int32                  ##Number of simulations performed
    nsuccesses::Int32             ##Number of successful simulations, excluding early rejections. Usually equal to nsims.
    parameters::Array{Float64, 2} ##parameters[i,j] is ith parameter for jth accepted sim
    sumstats::Array{Float64, 2}   ##sumstats[i,j] is ith sumstat for jth accepted sim
    distances::Array{Float64, 1}  ##distance[i] is distance for ith accepted sim
    weights::Array{Float64, 1}    ##weights[i] is weight for ith accepted sim
    abcdist::ABCDistance
    init_sims::Array{Float64, 2}  ##sims used for distance initialisation (only stored optionally)
    init_pars::Array{Float64, 2}  ##pars used for distance initialisation (only stored optionally)
end

##ABC PMC output
type ABCPMCOutput <: ABCOutput
    nparameters::Int32
    nsumstats::Int32  
    niterations::Int32            ##Number of iteration performed
    nsims::Int32                  ##Total number of simulations performed
    cusims::Array{Int32, 1}       ##cusims[i] is cumulative sims used up to end of iteration i
    parameters::Array{Float64, 3} ##parameters[i,j,k] is ith parameter for jth accepted sim in iteration k
    sumstats::Array{Float64, 3}   ##sumstats[i,j,k] is ith sumstat for jth accepted sim in iteration k
    distances::Array{Float64, 2}  ##distances[i,j] is distance for ith accepted sim in iteration j
    weights::Array{Float64, 2}    ##weights[i,j] is weight for ith accepted sim in iteration j
    abcdists::Array{ABCDistance, 1}    ##abcdist[i] is distance used in iteration i
    thresholds::Array{Float64, 1}  ##threshold[i] is threshold used in iteration i
    init_sims::Array{Array{Float64, 2}, 1} ##init_sims[i] is sims for distance initialisation at iteration i (only stored optionally)
    init_pars::Array{Array{Float64, 2}, 1} ##init_pars[i] is pars for distance initialisation at iteration i (only stored optionally)
end

##This needs a show function

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
        if (maximum(out.weights)==minimum(out.weights)) 
          (CI_lower[i], CI_upper[i]) = quantile(y, [0.025,0.975])
        else
          ##Crude way to approximate weighted quantiles
          z = sample(y, WeightVec(out.weights), 1000)
          (CI_lower[i], CI_upper[i]) = quantile(z, [0.025,0.975])
        end
    end
    print("ABC output, $k accepted values from $(out.nsims) simulations\n")
    ess = sum(out.weights)^2 / sum(out.weights.^2)
    print("Effective sample size $(round(ess,1))\n")
    print("Means and rough 95% credible intervals:\n")
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

##Return the parameter means (a vector)
function parameter_means(out::ABCRejOutput)
    vec(mean(out.parameters, WeightVec(out.weights), 2))
end

##Return parameter means in each iteration. The [i,j] entry is for parameter i in iteration j.
function parameter_means(out::ABCPMCOutput)
    means = Array(Float64, (out.nparameters, out.niterations))
    for (it in 1:out.niterations)
      means[:, it] = mean(out.parameters[:,:,it], WeightVec(out.weights[:,it]), 2)
    end
    means
end

##Return marginal parameter variances (a vector)
function parameter_vars(out::ABCRejOutput)
  vec(var(out.parameters, WeightVec(out.weights), 2))
end

##Return marginal parameter variances in each iteration. The [i,j] entry is for parameter i in iteration j.
function parameter_vars(out::ABCPMCOutput)
    vars = Array(Float64, (out.nparameters, out.niterations))
    for (it in 1:out.niterations)
      vars[:, it] = var(out.parameters[:,:,it], WeightVec(out.weights[:,it]), 2)
    end
    vars
end

##Return parameter covariance matrix
function parameter_covs(out::ABCRejOutput)
    cov(out.parameters, WeightVec(out.weights), vardim=2)
end

##Return parameter covariance matrix for each iteration. The [:,:,k] entry is for iteration k.
function parameter_covs(out::ABCPMCOutput)
    covs = Array(Float64, (out.nparameters, out.nparameters, out.niterations))
    for (it in 1:out.niterations)
      covs[:, :, it] = cov(out.parameters[:,:,it], WeightVec(out.weights[:,it]), vardim=2)
    end
    covs
end
