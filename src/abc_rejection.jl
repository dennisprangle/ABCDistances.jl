##############
##ABC-Rejection code
##############

##First stage of analysis (all simulations)
function RefTable(abcinput::ABCInput, nsims::Int32)
    x = abcinput
    parameters = Array(Float64, (x.nparameters,nsims))
    sumstats = Array(Float64, (x.nsumstats, nsims))
    for i=1:nsims
        pars = x.rprior()
        parameters[:,i] = pars
        sumstats[:,i] = x.data2sumstats(x.rdata(pars))
    end    
    RefTable(nsims,parameters,sumstats)
end

#=
## Trial parallel version
function RefTable(abcinput::ABCInput, nsims::Int32)
    x = abcinput
    ##temp = @parallel [x.rprior() for i in 1:nsims]
    ##parameters = hcat(temp...)
    temp = pmap(i->x.rprior(), 1:nsims)
    parameters = hcat(temp...)
    ##temp = @parallel [x.data2sumstats(x.rdata(parameters[:,i])) for i in 1:nsims]
    ##sumstats = hcat(temp...)
    temp = pmap(i->x.data2sumstats(x.rdata(parameters[:,i])), 1:nsims)
    sumstats = hcat(temp...)
    RefTable(nsims,parameters,sumstats)
end
=#

##Second stage of analysis (distances)
function ABCOutput(abcinput::ABCInput, abcnorm::ABCNorm, reftable::RefTable)
    x = abcinput
    r = reftable
    ##Calculate distances
    distances = [ evalnorm(abcnorm, abs(x.sobs-r.sumstats[:,i])) for i=1:r.nsims ]
    ABCOutput(r.nsims, r.parameters, r.sumstats, distances, ones(r.nsims), abcnorm)
end

##Sort output into distance order
function sortABCOutput!(abcoutput::ABCOutput)
    x = abcoutput
    ##Sort results into closeness order
    closenessorder = sortperm(x.distances)
    x.parameters = x.parameters[:,closenessorder]
    x.sumstats = x.sumstats[:,closenessorder]
    x.distances = x.distances[closenessorder]
    x.weights = x.weights[closenessorder]
    return
end

##Do simulations, calculate distances and sort output
function ABCOutput(abcinput::ABCInput, nsims::Integer)
    r = RefTable(abcinput, nsims)
    newnorm = init(abcinput.abcnorm, r.sumstats)
    abcoutput = ABCOutput(abcinput, newnorm, r)
    sortABCOutput!(abcoutput)
    abcoutput
end

##As above but return only k best matches
function ABCOutput(abcinput::ABCInput, nsims::Integer, k::Integer)
    abcoutput = ABCOutput(abcinput, nsims)
    abcoutput.parameters = abcoutput.parameters[:,1:k]
    abcoutput.sumstats = abcoutput.sumstats[:,1:k]
    abcoutput.distances = abcoutput.distances[1:k]
    abcoutput.weights = abcoutput.weights[1:k]
    abcoutput
end

##As above but return only matches with distance <= h
function ABCOutput(abcinput::ABCInput, nsims::Integer, h::FloatingPoint)
    abcoutput = ABCOutput(abcinput, nsims)
    if (abcoutput.distances[nsims] <= h)
        k = nsims
    else
        k = findfirst((x)->x>h, abcoutput.distances) - 1
    end
    abcoutput.parameters = abcoutput.parameters[:,1:k]
    abcoutput.sumstats = abcoutput.sumstats[:,1:k]
    abcoutput.distances = abcoutput.distances[1:k]
    abcoutput.weights = abcoutput.weights[1:k]
    abcoutput
end
