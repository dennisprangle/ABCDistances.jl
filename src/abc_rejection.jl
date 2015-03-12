##############
##ABC-Rejection code
##############

##Do abcRejection calculations but accept everything
##i.e. do simulations, calculate distances and sort into distance order
function abcRejection(in::ABCInput, nsims::Integer)
    parameters = Array(Float64, (in.nparameters,nsims))
    sumstats = zeros(Float64, (in.nsumstats, nsims))
    successes = Array(Bool, (nsims))
    for i in 1:nsims
        pars = in.rprior()
        parameters[:,i] = pars
        (success, stats) = in.sample_sumstats(pars)
        successes[i] = success
        if (success)
            sumstats[:,i] = stats
        end
    end
    nsuccesses = sum(successes)
    parameters = parameters[:, successes]
    sumstats = sumstats[:, successes]
    newdist = init(in.abcdist, sumstats)
    distances = [evaldist(newdist, sumstats[:,i]) for i=1:nsuccesses]
    out = ABCRejOutput(in.nparameters, in.nsumstats, nsims, nsuccesses, parameters, sumstats, distances, ones(nsims), newdist)
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
        k = out.nsuccesses
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
