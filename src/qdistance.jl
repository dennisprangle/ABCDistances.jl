##TO BE MERGED INTO distances.jl
##A quantile based distance
using Roots

type Qdistance <: ABCDistance
    sobs::Array{Float64, 1}
    w::Array{Float64, 1} ##Weights for each summary statistic
    p::Float64 ##Which quantile was used
end

##Assign arbitrary values to w and p prior to initialisation
function Qdistance(sobs::Array{Float64, 1})
    Qdistance(sobs, Array(Float64, 0), 0.5)
end

##FOLLOWING CODE EXACTLY THE SAME AS FOR MahalanobisDiag - DEFINE FOR A TYPE UNION?
function evaldist(x::Qdistance, s::Array{Float64, 1})
    absdiff = abs(x.sobs - s)
    norm(absdiff .* x.w, 2.0)
end

##sumstats should be array of all simulated summaries
##acceptable is array of those which passed the previous acceptance criteria
##k is how many should be accepted at this iteration
function init(x::Qdistance, sumstats::Array{Float64, 2}, acceptable::Array{Float64, 2}, k::Int32)
    (nstats, nsims) = size(sumstats)
    N = size(acceptable)[2]
    if (nsims == 0)
        ##TO DO: FIGURE OUT THE RIGHT THING TO RETURN IN THIS CASE
        error("sumstats contained no simulations")
    end
    absdiff_s = zeros(sumstats)
    absdiff_a = zeros(acceptable)
    for (i in 1:nstats)
        diffs_s = vec(sumstats[i,:]) .- x.sobs[i]
        absdiff_s[i,:] = sort!(abs(diffs_s))
        diffs_a = vec(acceptable[i,:]) .- x.sobs[i]
        absdiff_a[i,:] = sort!(abs(diffs_a))
    end
    function toroot(p::Float64)
        ##TO DO: this function may be discontinuous and not have a root if there are ties
        w = [1.0/quantile(vec(absdiff_s[i,:]), p) for i in 1:nstats] ##nb each call to quantile sorts the vector: inefficient
        d = [sum((absdiff_a[:,i] .* w).^2) for i in 1:N]
        sum(d .<= 1.0) - k
    end
    if toroot(1.0) < 0
        p = 1.0 ##If no p values meet criteria, choose max value
    else 
        p = fzero(toroot, 0.0, 1.0)
    end
    w = [1.0/quantile(vec(absdiff_s[i,:]), p) for i in 1:nstats] ##NB duplication of earlier code
    return Qdistance(x.sobs, w, p)
end

##Experimental version of abcSMC to work with Qdistance.
##MERGE THIS BACK IN TO MAIN CODE IF IT WORKS!
function qabcSMC(abcinput::ABCInput, N::Integer, k::Integer, maxsims::Integer, nsims_for_init=10000; adaptive=false, store_init=false)
    nparameters = length(abcinput.prior)
    ##First iteration is just standard rejection sampling
    curroutput = abcRejection(abcinput, N, k, store_init=store_init)
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
        newparameters = Array(Float64, (nparameters, N))
        newsumstats = Array(Float64, (abcinput.nsumstats, N))
        newpriorweights = Array(Float64, N)
        successes_thisit = 0
        if (adaptive || store_init)
            ##Initialise storage of all simulated summaries for use initialising the distance function
            sumstats_forinit = Array(Float64, (abcinput.nsumstats, nsims_for_init))
        end
        nextparticle = 1
        ##Loop to fill up new reference table
        while (nextparticle <= N && simsdone<maxsims)
            ##Sample parameters from importance density
            proppars = rimportance(curroutput, perturbdist)
            ##Calculate prior weight and reject if zero
            priorweight = pdf(abcinput.prior, proppars)
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
            if ((adaptive || store_init) && successes_thisit < nsims_for_init)
                successes_thisit += 1
                sumstats_forinit[:,successes_thisit] = propstats
            end
            if (adaptive)
                ##Accept if all prev distances less than corresponding thresholds.
                accept = propgood(propstats, dists, thresholds)
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
        ##Trim sumstats_forinit to correct size
        if (adaptive || store_init)
            if (successes_thisit < nsims_for_init)
                sumstats_forinit = sumstats_forinit[:,1:successes_thisit]
            end
        else
            sumstats_forinit = Array(Float64, (0,0))
        end
        ##Create new distance if needed
        if (adaptive)
            newdist = init(abcinput.abcdist, sumstats_forinit, newsumstats, k)
        else
            newdist = dists[1]
        end
        push!(dists, newdist)
        
        ##Calculate distances
        distances = [ evaldist(newdist, newsumstats[:,i]) for i=1:N ]
        oldoutput = copy(curroutput)
        curroutput = ABCRejOutput(nparameters, abcinput.nsumstats, N, N, newparameters, newsumstats, distances, zeros(N), newdist, sumstats_forinit) ##Set new weights to zero for now
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
    parameters = Array(Float64, (nparameters, k, itsdone))
    sumstats = Array(Float64, (abcinput.nsumstats, k, itsdone))
    distances = Array(Float64, (k, itsdone))
    weights = Array(Float64, (k, itsdone))
    for i in 1:itsdone        
        parameters[:,:,i] = rejOutputs[i].parameters
        sumstats[:,:,i] = rejOutputs[i].sumstats
        distances[:,i] = rejOutputs[i].distances
        weights[:,i] = rejOutputs[i].weights
    end
    if (store_init)
        init_sims = Array(Array{Float64, 2}, itsdone)
        for i in 1:itsdone
            init_sims[i] = rejOutputs[i].init_sims
        end
    else
        init_sims = Array(Array{Float64, 2}, 0)
    end
    output = ABCSMCOutput(nparameters, abcinput.nsumstats, itsdone, simsdone, cusims, parameters, sumstats, distances, weights, dists, thresholds, init_sims)
end
