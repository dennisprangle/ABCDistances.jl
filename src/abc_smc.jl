##k is how many particles to use
##nparticles defines the alpha quantile; nparticles=alpha*k
##maxsims - the algorithm will stop once this many simulations have been performed
##nsims_for_init - how many simulations to store to initialise the distance function
##adaptive - whether to use the adaptive or non-adaptive algorithm
function abcSMC(abcinput::ABCInput, nparticles::Integer, k::Integer, maxsims::Integer, nsims_for_init=10000; adaptive=false)
    ##First iteration is just standard rejection sampling
    curroutput = abcRejection(abcinput, k)
    itsdone = 1
    print("Iteration $itsdone, $nparticles sims done\n")
    print("Output of most recent stage:\n")
    print(curroutput)
    ##TO DO: Consider some stopping conditions? (e.g. threshold = 0) Call a "stopearly" method?
    ##We record a sequence of norms and thresholds
    ##(for non-adaptive case all norms the same, and we only use most recent threshold)
    norms = [curroutput.abcnorm]
    thresholds = [curroutput.distances[k]]
    rejOutputs = [curroutput]
    simsdone = k
    cusims = [k]
    ##Main loop
    while (simsdone < maxsims)
        ##Calculate variance of current weighted particle approximation
        currvar = cov(curroutput.parameters, WeightVec(curroutput.weights), vardim=2)
        perturbdist = MvNormal(2.0 .* currvar)
        ##Initialise new reference table
        newparameters = Array(Float64, (abcinput.nparameters, nparticles))
        newsumstats = Array(Float64, (abcinput.nsumstats, nparticles))
        newabsdiffs = Array(Float64, (abcinput.nsumstats, nparticles))
        if (adaptive)
            ##Initialise storage of all simulated summaries for use updating the distance function
            allsumstats = Array(Float64, (abcinput.nsumstats, nsims_for_init))
            simsdone_thisiteration = 0
        end
        nextparticle = 1
        ##Loop to fill up new reference table
        while (nextparticle <= nparticles && simsdone<maxsims)
            ##Sample parameters from importance density
            proppars = rimportance(curroutput, perturbdist)
            ##Draw summaries
            propstats = abcinput.sample_sumstats(proppars)
            absdiff = abs(abcinput.sobs-propstats)
            simsdone += 1
            if (adaptive && simsdone_thisiteration < nsims_for_init)
                simsdone_thisiteration += 1
                allsumstats[:,simsdone_thisiteration] = propstats
            end
            if (adaptive)
                ##Accept if all prev norms less than corresponding thresholds.
                accept = propgood(absdiff, norms, thresholds)
            else
                ##Accept if norm less than current threshold
                accept = propgood(absdiff, norms[itsdone], thresholds[itsdone])
            end
            if (accept)
                newparameters[:,nextparticle] = proppars
                newsumstats[:,nextparticle] = propstats
                newabsdiffs[:,nextparticle] = absdiff
                nextparticle += 1
            end
        end
        ##Stop if not all sims required to continue have been done (because simsdone==maxsims)
        if nextparticle<=nparticles
            continue
        end
        ##Update counters
        itsdone += 1
        push!(cusims, simsdone)
        ##Create new norm if needed
        if (adaptive)
            if (simsdone_thisiteration < nsims_for_init)
                allsumstats = allsumstats[:,1:simsdone_thisiteration]
            end
            newnorm = init(abcinput.abcnorm, allsumstats)
        else
            newnorm = norms[1]
        end
        push!(norms, newnorm)
        ##Calculate distances
        distances = [ evalnorm(newnorm, newabsdiffs[:,i]) for i=1:nparticles ]
        oldoutput = copy(curroutput)
        curroutput = ABCRejOutput(nparticles, newparameters, newsumstats, distances, zeros(nparticles), newnorm) ##Set new weights to zero for now
        sortABCOutput!(curroutput)
        ##Calculate, store and use new threshold
        newthreshold = curroutput.distances[k]
        push!(thresholds, newthreshold)
        curroutput.parameters = curroutput.parameters[:,1:k]
        curroutput.sumstats = curroutput.sumstats[:,1:k]
        curroutput.distances = curroutput.distances[1:k]
        curroutput.weights = getweights(curroutput, abcinput, oldoutput, perturbdist)
        ##Record output
        push!(rejOutputs, curroutput)
        ##Report status
        print("Iteration $itsdone, $simsdone sims done\n")
        print("Output of most recent stage:\n")
        print(curroutput)
        ##Make some plots as well?
        ##Consider alternative stopping conditions? (e.g. zero threshold reached)
    end
        
    ##Put results into ABCSMCOutput object
    parameters = Array(Float64, (abcinput.nparameters, k, itsdone))
    sumstats = Array(Float64, (abcinput.nsumstats, k, itsdone))
    distances = Array(Float64, (k, itsdone))
    weights = Array(Float64, (k, itsdone))
    for i in 1:itsdone        
        parameters[:,:,i] = rejOutputs[i].parameters
        sumstats[:,:,i] = rejOutputs[i].sumstats
        distances[:,i] = rejOutputs[i].distances
        weights[:,i] = rejOutputs[i].weights
    end
    output = ABCSMCOutput(itsdone, simsdone, cusims, parameters, sumstats, distances, weights, norms, thresholds)
end

##Check if summary statistics meet acceptance requirement
function propgood(absdiff::Array{Float64, 1}, norm::ABCNorm, threshold::Float64)
    return evalnorm(norm, absdiff)<=threshold
end

##Check if summary statistics meet all of previous acceptance requirements
function propgood(absdiff::Array{Float64, 1}, norms::Array, thresholds::Array{Float64, 1})
    ##TO DO: how to specify correct type for norms?
    for i in 1:length(norms)
        if !propgood(absdiff, norms[i], thresholds[i])
            return false
        end
    end
    return true
end

##Samples from importance density defined by prev output
function rimportance(out::ABCRejOutput, dist::MvNormal)
    i = sample(WeightVec(out.weights))
    out.parameters[:,i] + rand(dist)
end

##Calculate a single importance weight
function get1weight(x::Array{Float64,1}, in::ABCInput, old::ABCRejOutput, perturbdist::MvNormal)
    nparticles = size(old.parameters)[2]
    temp = [pdf(perturbdist, x-old.parameters[:,i]) for i in 1:nparticles]
    in.dprior(x) / sum(old.weights .* temp)
end

##Calculates importance weights
function getweights(current::ABCRejOutput, in::ABCInput, old::ABCRejOutput, perturbdist::MvNormal)
    nparticles = size(current.parameters)[2]
    weights = [get1weight(current.parameters[:,i], in, old, perturbdist) for i in 1:nparticles]
    weights ./ sum(weights)
end
