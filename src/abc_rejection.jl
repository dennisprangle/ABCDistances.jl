##############
##ABC-Rejection code
##############

##Do abcRejection calculations but accept everything
##i.e. do simulations, calculate distances and sort into distance order
function abcRejection(in::ABCInput, nsims::Integer)
    parameters = Array(Float64, (in.nparameters,nsims))
    sumstats = Array(Float64, (in.nsumstats, nsims))
    for i in 1:nsims
        pars = in.rprior()
        parameters[:,i] = pars
        sumstats[:,i] = in.data2sumstats(in.rdata(pars))
    end
    newnorm = init(in.abcnorm, sumstats)
    distances = [evalnorm(newnorm, abs(in.sobs-sumstats[:,i])) for i=1:nsims]
    out = ABCOutput(nsims, parameters, sumstats, distances, ones(nsims), newnorm)
    sortABCOutput!(out)
    out
end

##Do abcRejection, accepting k closest matches
function abcRejection(in::ABCInput, nsims::Integer, k::Integer)
    out = abcRejection(in, nsims)
    out.parameters = out.parameters[:,1:k]
    out.sumstats = out.sumstats[:,1:k]
    out.distances = out.distances[1:k]
    out.weights = out.weights[1:k]
    out
end

##Do abcRejection, accepting distances <= h
function abcRejection(in::ABCInput, nsims::Integer, h::FloatingPoint)
    out = abcRejection(in, nsims)
    if (out.distances[nsims] <= h)
        k = nsims
    else
        k = findfirst((x)->x>h, out.distances) - 1
    end
    ##nb the following works sensibly even if k==0
    out.parameters = out.parameters[:,1:k]
    out.sumstats = out.sumstats[:,1:k]
    out.distances = out.distances[1:k]
    out.weights = out.weights[1:k]
    out
end
