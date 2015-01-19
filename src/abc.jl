#####################################
##General ABC code
##i.e. type definitions and utilities
#####################################

##################
##TYPE DEFINITIONS
##################
##Input for an ABC analysis
type ABCInput
    rprior::Function
    ##dprior::Function ##TO DO: add this! Should be optional (default improper uniform?)
    rdata::Function
    data2sumstats::Function
    abcnorm::ABCNorm
    sobs::Array{Float64, 1}
    nparameters::Int32
    nsumstats::Int32    
    ##Maybe add constructor with sensible defaults?
end

##Partial results of ABC analysis
type RefTable
    nsims::Int32
    parameters::Array{Float64, 2}
    sumstats::Array{Float64, 2}
end

##Full results of ABC analysis
type ABCOutput
    nsims::Int32
    parameters::Array{Float64, 2}
    sumstats::Array{Float64, 2}
    distances::Array{Float64, 1}
    weights::Array{Float64, 1}
    abcnorm::ABCNorm
end

#################
##UTILITY METHODS
#################
function show(io::IO, x::ABCOutput)
    (p,k) = size(x.parameters)
    means = Array(Float64, p)
    CI_lower = Array(Float64, p)
    CI_upper = Array(Float64, p)
    for i in 1:p
        y = squeeze(x.parameters[i,:], 1)
        means[i] = sum(y.*x.weights)/sum(x.weights)
        if (maximum(x.weights)==minimum(x.weights)) 
          (CI_lower[i], CI_upper[i]) = quantile(y, [0.025,0.975])
        else
          ##Crude way to approximate weighted quantiles
          z = sample(y, WeightVec(x.weights), 1000)
          (CI_lower[i], CI_upper[i]) = quantile(z, [0.025,0.975])
        end
    end
    print("ABC output, $k accepted values from $(x.nsims) simulations\n")
    print("Means and rough 95% credible intervals:\n")
    for (i in 1:p)
        @printf("Parameter %d: %.2e (%.2e,%.2e)\n", i, means[i], CI_lower[i], CI_upper[i])
    end
end
