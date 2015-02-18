##k is how many particles to use
##nparticles defines the alpha quantile; nparticles=alpha*k
##maxsims - the algorithm will stop once this many simulations have been performed
##nsims_for_init - how many simulations to store to initialise the distance function
##adaptive - whether to use the adaptive or non-adaptive algorithm
function abcSMC(abcinput::ABCInput, nparticles::Integer, k::Integer, maxsims::Integer, nsims_for_init=10000; adaptive=false)
    iteration = 1
    ##First iteration is just standard rejection sampling
    curroutput = abcRejection(abcinput, nparticles)
    ##TO DO: Consider some stopping conditions? (e.g. threshold = 0) Call a "stopearly" method?
    print("Iteration $iteration, $nparticles sims done\n")
    print("Output of most recent stage:\n")
    print(curroutput)
    ##For the adaptive algorithm there will be a sequence of norms and thresholds
    ##In the non-adaptive case norms and thresholds will stay at length 1
    norms = [curroutput.abcnorm]
    thresholds = [curroutput.distances[k]]
    output = [curroutput]
    simsdone = nparticles
    ##Main loop
    while (simsdone < maxsims)
        iteration += 1
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
            ##Accept if all prev norms less than corresponding thresholds.
            if propgood(absdiff, norms, thresholds)
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
        if (adaptive)
            if (simsdone_thisiteration < nsims_for_init)
                allsumstats = allsumstats[:,1:simsdone_thisiteration]
            end
            ##Initialise new norm
            newnorm = init(abcinput.abcnorm, allsumstats)
            push!(norms, newnorm)
        else
            newnorm = norms[1]
        end
        ##Calculate distances
        distances = [ evalnorm(newnorm, newabsdiffs[:,i]) for i=1:nparticles ]
        oldoutput = copy(curroutput)
        curroutput = ABCRejOutput(nparticles, newparameters, newsumstats, distances, zeros(nparticles), newnorm) ##Set new weights to zero for now
        sortABCOutput!(curroutput)
        ##Calculate, store and use new threshold
        newthreshold = curroutput.distances[k]
        if (adaptive) 
            push!(thresholds, newthreshold)
        else
            thresholds[1] = newthreshold
        end
        curroutput.parameters = curroutput.parameters[:,1:k]
        curroutput.sumstats = curroutput.sumstats[:,1:k]
        curroutput.distances = curroutput.distances[1:k]
        curroutput.weights = getweights(curroutput, abcinput, oldoutput, perturbdist)
        ##Record output
        push!(output, curroutput)
        ##Report status
        print("Iteration $iteration, $simsdone sims done\n")
        print("Output of most recent stage:\n")
        print(curroutput)
        ##Make some plots as well?
        ##Consider alternative stopping conditions? (e.g. zero threshold reached)
    end
    output
end

##Check if summary statistics meet all of previous acceptance requirements
function propgood(absdiff::Array{Float64, 1}, norms::Array, thresholds::Array{Float64, 1})
    ##TO DO: how to specify correct type for norms?
    for i in 1:length(norms)
        if evalnorm(norms[i], absdiff)>thresholds[i]
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
