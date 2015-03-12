##k is how many particles to use
##N is how many acceptances to require at each iteration
##maxsims - the algorithm will stop once this many simulations have been performed
##nsims_for_init - how many simulations to store to initialise the distance function
##adaptive - whether to use the adaptive or non-adaptive algorithm
function abcSMC(abcinput::ABCInput, N::Integer, k::Integer, maxsims::Integer, nsims_for_init=10000; adaptive=false)
    ##First iteration is just standard rejection sampling
    curroutput = abcRejection(abcinput, N, k)
    itsdone = 1
    print("Iteration $itsdone, $N sims done\n")
    @printf("Acceptance rate %.1e percent\n", 100*k/N)
    print("Output of most recent stage:\n")
    print(curroutput)
    ##TO DO: Consider some stopping conditions? (e.g. threshold = 0) Call a "stopearly" method?
    ##We record a sequence of distances and thresholds
    ##(for non-adaptive case all distances the same, and we only use most recent threshold)
    dists = [curroutput.abcdist]
    thresholds = [curroutput.distances[k]]
    rejOutputs = [curroutput]
    simsdone = N
    cusims = [N]
    ##Main loop
    while (simsdone < maxsims)
        ##Calculate variance of current weighted particle approximation
        currvar = cov(curroutput.parameters, WeightVec(curroutput.weights), vardim=2)
        perturbdist = MvNormal(2.0 .* currvar)
        ##Initialise new reference table
        newparameters = Array(Float64, (abcinput.nparameters, N))
        newsumstats = Array(Float64, (abcinput.nsumstats, N))
        newpriorweights = Array(Float64, N)
        successes_thisit = 0
        if (adaptive)
            ##Initialise storage of all simulated summaries for use initialising the distance function
            sumstats_forinit = Array(Float64, (abcinput.nsumstats, nsims_for_init))
            acceptable = Array(Bool, nsims_for_init)
        end
        nextparticle = 1
        ##Loop to fill up new reference table
        while (nextparticle <= N && simsdone<maxsims)
            ##Sample parameters from importance density
            proppars = rimportance(curroutput, perturbdist)
            ##Calculate prior weight and reject if zero
            priorweight = abcinput.dprior(proppars)
            if (priorweight == 0.0)
                continue
            end           
            ##Draw summaries
            (success, propstats) = abcinput.sample_sumstats(proppars)
            simsdone += 1
            if (!success)
                ##If rejection occurred during simulation
                continue
            end
            if (adaptive && successes_thisit < nsims_for_init)
                successes_thisit += 1
                sumstats_forinit[:,successes_thisit] = propstats
            end
            if (adaptive)
                ##Accept if all prev distances less than corresponding thresholds.
                accept = propgood(propstats, dists, thresholds)
                acceptable[successes_thisit] = accept
            else
                ##Accept if distance less than current threshold
                accept = propgood(propstats, dists[itsdone], thresholds[itsdone])
            end
            if (accept)
                newparameters[:,nextparticle] = proppars
                newsumstats[:,nextparticle] = propstats
                newpriorweights[nextparticle] = priorweight
                nextparticle += 1
            end
        end
        ##Stop if not all sims required to continue have been done (because simsdone==maxsims)
        if nextparticle<=N
            continue
        end
        ##Update counters
        itsdone += 1
        push!(cusims, simsdone)
        ##Create new distance if needed
        if (adaptive)
            if (successes_thisit < nsims_for_init)
                sumstats_forinit = sumstats_forinit[:,1:successes_thisit]
                acceptable = acceptable[:,1:successes_thisit]
            end
            newdist = init(abcinput.abcdist, sumstats_forinit, acceptable)
        else
            newdist = dists[1]
        end
        push!(dists, newdist)
        ##Calculate distances
        distances = [ evaldist(newdist, newsumstats[:,i]) for i=1:N ]
        oldoutput = copy(curroutput)
        curroutput = ABCRejOutput(abcinput.nparameters, abcinput.nsumstats, N, N, newparameters, newsumstats, distances, zeros(N), newdist) ##Set new weights to zero for now
        sortABCOutput!(curroutput)
        ##Calculate, store and use new threshold
        newthreshold = curroutput.distances[k]
        push!(thresholds, newthreshold)
        curroutput.parameters = curroutput.parameters[:,1:k]
        curroutput.sumstats = curroutput.sumstats[:,1:k]
        curroutput.distances = curroutput.distances[1:k]
        curroutput.weights = getweights(curroutput, newpriorweights, oldoutput, perturbdist)
        ##Record output
        push!(rejOutputs, curroutput)
        ##Report status
        print("Iteration $itsdone, $simsdone sims done\n")
        @printf("Acceptance rate %.1e percent\n", 100*k/(cusims[itsdone]-cusims[itsdone-1]))
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
    output = ABCSMCOutput(abcinput.nparameters, abcinput.nsumstats, itsdone, simsdone, cusims, parameters, sumstats, distances, weights, dists, thresholds)
end

##Check if summary statistics meet acceptance requirement
function propgood(s::Array{Float64, 1}, dist::ABCDistance, threshold::Float64)
    return evaldist(dist, s)<=threshold
end

##Check if summary statistics meet all of previous acceptance requirements
function propgood(s::Array{Float64, 1}, dists::Array, thresholds::Array{Float64, 1})
    for i in [length(dists):-1:1] ##Check the most stringent case first
        if !propgood(s, dists[i], thresholds[i])
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
function get1weight(x::Array{Float64,1}, priorweight::Float64, old::ABCRejOutput, perturbdist::MvNormal)
    nparticles = size(old.parameters)[2]
    temp = [pdf(perturbdist, x-old.parameters[:,i]) for i in 1:nparticles]
    priorweight / sum(old.weights .* temp)
end

##Calculates importance weights
function getweights(current::ABCRejOutput, priorweights::Array{Float64,1}, old::ABCRejOutput, perturbdist::MvNormal)
    nparticles = size(current.parameters)[2]
    weights = [get1weight(current.parameters[:,i], priorweights[i], old, perturbdist) for i in 1:nparticles]
    weights ./ sum(weights)
end
