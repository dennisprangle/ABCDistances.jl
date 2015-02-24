#####################################
##General ABC code
##i.e. type definitions and utilities
#####################################

##################
##TYPE DEFINITIONS
#################
##Input for an ABC analysis
type ABCInput
    rprior::Function
    dprior::Function
    sample_sumstats::Function
    abcnorm::ABCNorm
    sobs::Array{Float64, 1}
    nparameters::Int32
    nsumstats::Int32    
end

##Full results of an ABC analysis
abstract ABCOutput

##Rejection sampling output
type ABCRejOutput <: ABCOutput
    nparameters::Int32
    nsumstats::Int32 
    nsims::Int32
    parameters::Array{Float64, 2} ##parameters[i,j] is ith parameter for jth accepted sim
    sumstats::Array{Float64, 2}   ##sumstats[i,j] is ith sumstat for jth accepted sim
    distances::Array{Float64, 1}  ##distance[i] is distance for ith accepted sim
    weights::Array{Float64, 1}    ##weights[i] is weight for ith accepted sim
    abcnorm::ABCNorm
end

##ABC SMC output
type ABCSMCOutput <: ABCOutput
    nparameters::Int32
    nsumstats::Int32  
    niterations::Int32            ##Number of iteration performed
    nsims::Int32                  ##Total number of simulations performed
    cusims::Array{Int32, 1}       ##cusims[i] is cumulative sims used up to end of iteration i
    parameters::Array{Float64, 3} ##parameters[i,j,k] is ith parameter for jth accepted sim in iteration k
    sumstats::Array{Float64, 3}   ##sumstats[i,j,k] is ith sumstat for jth accepted sim in iteration k
    distances::Array{Float64, 2}  ##distances[i,j] is distance for ith accepted sim in iteration j
    weights::Array{Float64, 2}    ##weights[i,j] is weight for ith accepted sim in iteration j
    abcnorms::Array{ABCNorm, 1}    ##abcnorm[i] is norm used in iteration i
    thresholds::Array{Float64, 1}  ##threshold[i] is threshold used in iteration i
end

##This needs a show function
##Also want to be able to calculate parameter means and variances for both types of output. We should be able to get marginal variances or matrix.

#################
##CONSTRUCTORS
#################
##Semi-sensible defaults
function ABCInput()
    ABCInput(()->rand(1),  ##rprior U(0,1) prior on 1 parameter
             (x)->1.0,      ##dprior is improper uniform prior
             (x)->rand(1), ##sample_sumstats draws from U(0,1) independent of parameters
             Euclidean(),  ##abcnorm
             [1.0],         ##sobs
             1,             ##nparameters
             1)             ##nsumstats
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
    print("Means and rough 95% credible intervals:\n")
    for (i in 1:p)
        @printf("Parameter %d: %.2e (%.2e,%.2e)\n", i, means[i], CI_lower[i], CI_upper[i])
    end
end

function copy(out::ABCRejOutput)
    ABCRejOutput(out.nparameters, out.nsumstats, out.nsims, out.parameters, out.sumstats, out.distances, out.weights, out.abcnorm)
end

##Sort output into distance order
function sortABCOutput!(out::ABCRejOutput)
    ##Sort results into closeness order
    closenessorder = sortperm(out.distances)
    out.parameters = out.parameters[:,closenessorder]
    out.sumstats = out.sumstats[:,closenessorder]
    out.distances = out.distances[closenessorder]
    out.weights = out.weights[closenessorder]
    return
end

##Return the parameter means (a vector)
function parameter_means(out::ABCRejOutput)
    mean(out.parameters, WeightVec(out.weights), 2)
end

##Return parameter means in each iteration. The [i,j] entry is for parameter i in iteration j.
function parameter_means(out::ABCSMCOutput)
    means = Array(Float64, (out.nparameters, out.niterations))
    for (it in 1:out.niterations)
      means[:, it] = mean(out.parameters[:,:,it], WeightVec(out.weights[:,it]), 2)
    end
    means
end

##Return marginal parameter variances (a vector)
function parameter_vars(out::ABCRejOutput)
  var(out.parameters, WeightVec(out.weights), 2)
end

##Return marginal parameter variances in each iteration. The [i,j] entry is for parameter i in iteration j.
function parameter_vars(out::ABCSMCOutput)
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
function parameter_covs(out::ABCSMCOutput)
    covs = Array(Float64, (out.nparameters, out.nparameters, out.niterations))
    for (it in 1:out.niterations)
      covs[:, :, it] = cov(out.parameters[:,:,it], WeightVec(out.weights[:,it]), vardim=2)
    end
    covs
end
