######################################################
##The ABC-PMC algorithms and functions they all share.
##See abc_pmc_interace.jl for interface to most of these
##(The exception is Algorithm 4 which doesn't require an interace).
######################################################

"
Perform a version of ABC-PMC proposed in the paper, Algorithm 4.
This chooses the distance function for iteration t at the end of iteration t-1.

Its arguments are as follows:

* `abcinput`
An `ABCInput` variable.

* `N`
Number of accepted particles in each iteration.

* `α`
A tuning parameter between 0 and 1 determining how fast the acceptance threshold is reduced. (In more detai, the acceptance threshold in iteration t is the α quantile of distances from particles which meet the acceptance criteria of the previous iteration.)

* `maxsims`
The algorithm terminates once this many simulations have been performed.

* `nsims_for_init`
How many simulations are stored to initialise the distance function (by default 10,000).

* `store_init` (optional)
Whether to store the parameters and simulations used to initialise the distance function in each iteration (useful to produce many of the paper's figures).
These are stored for every iteration even if `adaptive` is false.
By default false.

* `diag_perturb` (optional)
Whether perturbation should be based on a diagonalised estimate of the covariance.
By default this is false, giving the perturbation described in Section 2.2 of the paper.

* `silent` (optional)
When true no progress bar or messages are shown during execution.
By default false.

* `h1` (optional)
The acceptance threshold for the first iteration, by default `Inf` (accept everything).
If this is not `Inf`, then `abcinput.abcdist` will be used in the first iteration, and so must not require initialisation or be pre-initialised.

The output is a `ABCPMCOutput` object.
"
function abcPMC4(abcinput::ABCInput, N::Integer, α::Float64, maxsims::Integer, nsims_for_init=10000; store_init=false, diag_perturb=false, silent=false, h1=Inf)
    if !silent
        prog = Progress(maxsims, 1) ##Progress meter
    end
    k::Int = ceil(N*α)
    nparameters = length(abcinput.prior)
    itsdone = 0
    simsdone = 0
    firstit = true
    ##We record a sequence of distances and thresholds
    ##(all distances the same but we record a sequence for consistency with other algorithm)
    dists = ABCDistance[abcinput.abcdist]
    thresholds = Float64[h1]
    rejOutputs = ABCRejOutput[]
    cusims = Int[]
    ##Main loop
    while (simsdone < maxsims)
        samplefromprior = (firstit || thresholds[itsdone]==Inf)
        if !samplefromprior
            perturbdist = getperturbdist(curroutput, diag_perturb)
        end
        ##Initialise new reference table
        newparameters = Array{Float64}(nparameters, N)
        newsumstats = Array{Float64}(abcinput.nsumstats, N)
        newpriorweights = Array{Float64}(N)
        successes_thisit = 0            
        nextparticle = 1
        ##Initialise storage of simulated parameter/summary pairs
        sumstats_forinit = Array{Float64}(abcinput.nsumstats, nsims_for_init)
        pars_forinit = Array{Float64}(nparameters, nsims_for_init)
        ##Loop to fill up new reference table
        while (nextparticle <= N && simsdone<maxsims)
            ##Sample parameters from importance density
            if samplefromprior
                proppars = rand(abcinput.prior)
            else
                proppars = rimportance(curroutput, perturbdist)
            end
            ##Calculate prior weight and reject if zero
            priorweight = pdf(abcinput.prior, proppars)
            if (priorweight == 0.0)
                continue
            end          
            ##Draw summaries
            (success, propstats) = abcinput.sample_sumstats(proppars)
            simsdone += 1
            if !silent
                next!(prog)
            end
            if (!success)
                ##If rejection occurred during simulation
                continue
            end
            if (successes_thisit < nsims_for_init)
                successes_thisit += 1
                sumstats_forinit[:,successes_thisit] = propstats
                pars_forinit[:,successes_thisit] = proppars
            end
            ##Accept if all previous distances less than corresponding thresholds
            accept = propgood(propstats, dists, thresholds)
            if (accept)
                newparameters[:,nextparticle] = copy(proppars)
                newsumstats[:,nextparticle] = copy(propstats)                
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
        ##Trim pars_forinit and sumstats_forinit to correct size
        if (successes_thisit < nsims_for_init)
            sumstats_forinit = sumstats_forinit[:,1:successes_thisit]
            pars_forinit = pars_forinit[:,1:successes_thisit]           
        end

        currdist = dists[itsdone]
        if firstit && h1==Inf
            currdistances = zeros(N)
        else
            currdistances = Float64[ evaldist(currdist, newsumstats[:,i]) for i in 1:N ]
        end
        if samplefromprior
            newweights = ones(N)
        else
            oldoutput = copy(curroutput)
            newweights = getweights(newparameters, newpriorweights, oldoutput, perturbdist)
        end
        curroutput = ABCRejOutput(nparameters, abcinput.nsumstats, N, N, newparameters, newsumstats, currdistances, newweights, currdist, sumstats_forinit, pars_forinit) ##n.b. abcinput.abcdist may be uninitialised, but this is not problematic

        ##Create and store new distance
        newdist = init(abcinput.abcdist, sumstats_forinit, pars_forinit)
        push!(dists, newdist)
            
        ##Calculate and store threshold for next iteration
        newdistances = Float64[ evaldist(newdist, newsumstats[:,i]) for i in 1:N ]
        newthreshold = select!(newdistances, k)
        push!(thresholds, newthreshold)
            
        ##Record output
        push!(rejOutputs, curroutput)
        ##Report status
        if !silent
            print("\n Iteration $itsdone, $simsdone sims done\n")
            if firstit
                accrate = k/simsdone            
            else
            accrate = k/(simsdone-cusims[itsdone-1])
            end
            @printf("Acceptance rate %.1e percent\n", 100*accrate)
            print("Output of most recent stage:\n")
            print(curroutput)
            print("Next threshold: $(convert(Float32, newthreshold))\n") ##Float64 shows too many significant figures
            ##TO DO: make some plots as well?
        end
        ##TO DO: consider alternative stopping conditions? (e.g. zero threshold reached)
        firstit = false
    end
        
    ##Put results into ABCPMCOutput object
    parameters = Array{Float64}(nparameters, N, itsdone)
    sumstats = Array{Float64}(abcinput.nsumstats, N, itsdone)
    distances = Array{Float64}(N, itsdone)
    weights = Array{Float64}(N, itsdone)
    for i in 1:itsdone        
        parameters[:,:,i] = rejOutputs[i].parameters
        sumstats[:,:,i] = rejOutputs[i].sumstats
        distances[:,i] = rejOutputs[i].distances
        weights[:,i] = rejOutputs[i].weights
    end
    if (store_init)
        init_sims = Array{Array{Float64, 2}}(itsdone)
        init_pars = Array{Array{Float64, 2}}(itsdone)
        for i in 1:itsdone
            init_sims[i] = rejOutputs[i].init_sims
            init_pars[i] = rejOutputs[i].init_pars
        end
    else
        init_sims = Array{Array{Float64, 2}}(0)
        init_pars = Array{Array{Float64, 2}}(0)
    end
    output = ABCPMCOutput(nparameters, abcinput.nsumstats, itsdone, simsdone, cusims, parameters, sumstats, distances, weights, dists[1:itsdone], thresholds[1:itsdone], init_sims, init_pars)
end

"
Perform a version of ABC-PMC: either Algorithm 5 or a variant of Algorithm 3.
See `abcPMC3V` and `abcPMC5` documentation for details.
The `adaptive` argument determines which algorithm is used (`true` gives Algorithm 5).
"
function abcPMC_3V5(abcinput::ABCInput, N::Integer, α::Float64, maxsims::Integer, adaptive::Bool, nsims_for_init=10000; store_init=false, diag_perturb=false, silent=false)
    if !silent
        prog = Progress(maxsims, 1) ##Progress meter
    end
    M::Int = ceil(N/α)
    nparameters = length(abcinput.prior)
    itsdone = 0
    simsdone = 0
    firstit = true
    ##We record a sequence of distances and thresholds
    ##(for non-adaptive case all distances the same, and we only use most recent threshold)
    dists = ABCDistance[]
    thresholds = Float64[]
    rejOutputs = ABCRejOutput[]
    cusims = Int[]
    ##Main loop
    while (simsdone < maxsims)
        samplefromprior = firstit
        if !samplefromprior
            perturbdist = getperturbdist(curroutput, diag_perturb)
        end
        ##Initialise new reference table
        newparameters = Array{Float64}(nparameters, M)
        newsumstats = Array{Float64}(abcinput.nsumstats, M)
        newpriorweights = Array{Float64}(M)
        successes_thisit = 0
        if (firstit || adaptive || store_init)
            ##Initialise storage of simulated parameter/summary pairs
            sumstats_forinit = Array{Float64}(abcinput.nsumstats, nsims_for_init)
            pars_forinit = Array{Float64}(nparameters, nsims_for_init)
        end
        nextparticle = 1
        ##Loop to fill up new reference table
        while (nextparticle <= M && simsdone<maxsims)
            ##Sample parameters from importance density
            if (firstit)
                proppars = rand(abcinput.prior)
            else
                proppars = rimportance(curroutput, perturbdist)
            end
            ##Calculate prior weight and reject if zero
            priorweight = pdf(abcinput.prior, proppars)
            if (priorweight == 0.0)
                continue
            end          
            ##Draw summaries
            (success, propstats) = abcinput.sample_sumstats(proppars)
            simsdone += 1
            if !silent
                next!(prog)
            end
            if (!success)
                ##If rejection occurred during simulation
                continue
            end
            if ((firstit || adaptive || store_init) && successes_thisit < nsims_for_init)
                successes_thisit += 1
                sumstats_forinit[:,successes_thisit] = propstats
                pars_forinit[:,successes_thisit] = proppars
            end
            if (firstit)
                ##No rejection at this stage in first iteration
                accept = true
            elseif (adaptive)
                ##Accept if all prev distances less than corresponding thresholds
                accept = propgood(propstats, dists, thresholds)
            else
                ##Accept if distance less than current threshold
                accept = propgood(propstats, dists[itsdone], thresholds[itsdone])
            end
            if (accept)
                newparameters[:,nextparticle] = copy(proppars)
                newsumstats[:,nextparticle] = copy(propstats)                
                newpriorweights[nextparticle] = priorweight
                nextparticle += 1
            end
        end
        ##Stop if not all sims required to continue have been done (because simsdone==maxsims)
        if nextparticle<=M
            continue
        end
        ##Update counters
        itsdone += 1
        push!(cusims, simsdone)
        ##Trim pars_forinit and sumstats_forinit to correct size
        if (firstit || adaptive || store_init)
            if (successes_thisit < nsims_for_init)
                sumstats_forinit = sumstats_forinit[:,1:successes_thisit]
                pars_forinit = pars_forinit[:,1:successes_thisit]
            end
        else
            ##Create some empty arrays to use as arguments
            sumstats_forinit = Array{Float64}(0,0)
            pars_forinit = Array{Float64}(0,0)
        end
        ##Create new distance if needed
        if (firstit || adaptive)
            newdist = init(abcinput.abcdist, sumstats_forinit, pars_forinit)
        else
            newdist = dists[1]
        end
        push!(dists, newdist)
        
        ##Calculate distances
        distances = Float64[ evaldist(newdist, newsumstats[:,i]) for i=1:M ]
        if !samplefromprior
            oldoutput = copy(curroutput)
        end
        curroutput = ABCRejOutput(nparameters, abcinput.nsumstats, M, N, newparameters, newsumstats, distances, newpriorweights, newdist, sumstats_forinit, pars_forinit) ##Temporarily use prior weights
        sortABCOutput!(curroutput)
        ##Calculate, store and use new threshold
        newthreshold = curroutput.distances[N]
        push!(thresholds, newthreshold)
        curroutput.parameters = curroutput.parameters[:,1:N]
        curroutput.sumstats = curroutput.sumstats[:,1:N]
        curroutput.distances = curroutput.distances[1:N]
        if samplefromprior
            curroutput.weights = ones(N)
        else
            curroutput.weights = getweights(curroutput.parameters, curroutput.weights, oldoutput, perturbdist)
        end
            
        ##Record output
        push!(rejOutputs, curroutput)
        ##Report status
        if !silent
            print("\n Iteration $itsdone, $simsdone sims done\n")
            if firstit
                accrate = N/simsdone            
            else
            accrate = N/(simsdone-cusims[itsdone-1])
            end
            @printf("Acceptance rate %.1e percent\n", 100*accrate)
            print("Output of most recent stage:\n")
            print(curroutput)
            ##TO DO: make some plots as well?
        end
        ##TO DO: consider alternative stopping conditions? (e.g. zero threshold reached)
        firstit = false
    end
        
    ##Put results into ABCPMCOutput object
    parameters = Array{Float64}(nparameters, N, itsdone)
    sumstats = Array{Float64}(abcinput.nsumstats, N, itsdone)
    distances = Array{Float64}(N, itsdone)
    weights = Array{Float64}(N, itsdone)
    for i in 1:itsdone        
        parameters[:,:,i] = rejOutputs[i].parameters
        sumstats[:,:,i] = rejOutputs[i].sumstats
        distances[:,i] = rejOutputs[i].distances
        weights[:,i] = rejOutputs[i].weights
    end
    if (store_init)
        init_sims = Array{Array{Float64, 2}}(itsdone)
        init_pars = Array{Array{Float64, 2}}(itsdone)
        for i in 1:itsdone
            init_sims[i] = rejOutputs[i].init_sims
            init_pars[i] = rejOutputs[i].init_pars
        end
    else
        init_sims = Array{Array{Float64, 2}}(0)
        init_pars = Array{Array{Float64, 2}}(0)
    end
    output = ABCPMCOutput(nparameters, abcinput.nsumstats, itsdone, simsdone, cusims, parameters, sumstats, distances, weights, dists, thresholds, init_sims, init_pars)
end

"
Perform a version of ABC-PMC: either Algorithm 2 or 3.
See `abcPMC2` and `abcPMC3` documentation for details.
The `initialise_dist` argument specifies whether the distance will be initialised at the end of the first iteration.
If `true` the code implements Algorithm 3.
If `false` a distance must be specified which does not need initialisation and Algorithm 2 is run.
"   
function abcPMC_23(abcinput::ABCInput, N::Integer, α::Float64, maxsims::Integer, initialise_dist::Bool, nsims_for_init=10000; store_init=false, diag_perturb=false, silent=false, h1=Inf)
    if initialise_dist && h1<Inf
        error("To initialise distance during the algorithm the first threshold must be Inf")
    end
    if !silent
        prog = Progress(maxsims, 1) ##Progress meter
    end
    k::Int = ceil(N*α)
    nparameters = length(abcinput.prior)
    itsdone = 0
    simsdone = 0
    firstit = true
    ##We record a sequence of distances and thresholds
    ##(all distances the same but we record a sequence for consistency with other algorithm)
    dists = ABCDistance[abcinput.abcdist]
    thresholds = Float64[h1]
    rejOutputs = ABCRejOutput[]
    cusims = Int[]
    ##Main loop
    while (simsdone < maxsims)
        samplefromprior = (firstit || thresholds[itsdone]==Inf)
        if !samplefromprior
            perturbdist = getperturbdist(curroutput, diag_perturb)
        end
        ##Initialise new reference table
        newparameters = Array{Float64}(nparameters, N)
        newsumstats = Array{Float64}(abcinput.nsumstats, N)
        newpriorweights = Array{Float64}(N)
        successes_thisit = 0
        if (firstit || store_init)
            ##Initialise storage of simulated parameter/summary pairs
            sumstats_forinit = Array{Float64}(abcinput.nsumstats, nsims_for_init)
            pars_forinit = Array{Float64}(nparameters, nsims_for_init)
        end
        nextparticle = 1
        ##Loop to fill up new reference table
        while (nextparticle <= N && simsdone<maxsims)
            ##Sample parameters from importance density
            if samplefromprior
                proppars = rand(abcinput.prior)
            else
                proppars = rimportance(curroutput, perturbdist)
            end
            ##Calculate prior weight and reject if zero
            priorweight = pdf(abcinput.prior, proppars)
            if (priorweight == 0.0)
                continue
            end          
            ##Draw summaries
            (success, propstats) = abcinput.sample_sumstats(proppars)
            simsdone += 1
            if !silent
                next!(prog)
            end
            if (!success)
                ##If rejection occurred during simulation
                continue
            end
            if (((firstit && initialise_dist) || store_init) && successes_thisit < nsims_for_init)
                successes_thisit += 1
                sumstats_forinit[:,successes_thisit] = propstats
                pars_forinit[:,successes_thisit] = proppars
            end
            if (firstit && initialise_dist)
                ##No rejection at this stage in first iteration if we want to initialise distance
                accept = true
            else
                ##Accept if distance less than current threshold
                accept = propgood(propstats, dists[1], thresholds[itsdone+1])
            end
            if (accept)
                newparameters[:,nextparticle] = copy(proppars)
                newsumstats[:,nextparticle] = copy(propstats)                
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
        ##Trim pars_forinit and sumstats_forinit to correct size
        if ((firstit && initialise_dist) || store_init)
            if (successes_thisit < nsims_for_init)
                sumstats_forinit = sumstats_forinit[:,1:successes_thisit]
                pars_forinit = pars_forinit[:,1:successes_thisit]
            end
        end
        ##Create new distance if needed
        if (firstit && initialise_dist)
            newdist = init(dists[1], sumstats_forinit, pars_forinit)
        else
            newdist = dists[1]
        end
        ##Store new distance
        if firstit
            dists[1] = newdist
        else
            push!(dists, newdist)
        end
        
        ##Calculate distances
        distances = Float64[ evaldist(newdist, newsumstats[:,i]) for i=1:N ]
        if !samplefromprior
            oldoutput = copy(curroutput)
        end
        curroutput = ABCRejOutput(nparameters, abcinput.nsumstats, N, N, newparameters, newsumstats, distances, newpriorweights, newdist, sumstats_forinit, pars_forinit) ##Temporarily use prior weights
        sortABCOutput!(curroutput)
        ##Calculate and store threshold for next iteration
        newthreshold = curroutput.distances[k]
        push!(thresholds, newthreshold)
        if samplefromprior
            curroutput.weights = ones(N)
        else
            curroutput.weights = getweights(curroutput.parameters, curroutput.weights, oldoutput, perturbdist)
        end
            
        ##Record output
        push!(rejOutputs, curroutput)
        ##Report status
        if !silent
            print("\n Iteration $itsdone, $simsdone sims done\n")
            if firstit
                accrate = k/simsdone            
            else
            accrate = k/(simsdone-cusims[itsdone-1])
            end
            @printf("Acceptance rate %.1e percent\n", 100*accrate)
            print("Output of most recent stage:\n")
            print(curroutput)
            print("Next threshold: $(convert(Float32, newthreshold))\n") ##Float64 shows too many significant figures
            ##TO DO: make some plots as well?
        end
        ##TO DO: consider alternative stopping conditions? (e.g. zero threshold reached)
        firstit = false
    end
        
    ##Put results into ABCPMCOutput object
    parameters = Array{Float64}(nparameters, N, itsdone)
    sumstats = Array{Float64}(abcinput.nsumstats, N, itsdone)
    distances = Array{Float64}(N, itsdone)
    weights = Array{Float64}(N, itsdone)
    for i in 1:itsdone        
        parameters[:,:,i] = rejOutputs[i].parameters
        sumstats[:,:,i] = rejOutputs[i].sumstats
        distances[:,i] = rejOutputs[i].distances
        weights[:,i] = rejOutputs[i].weights
    end
    if (store_init)
        init_sims = Array{Array{Float64, 2}}(itsdone)
        init_pars = Array{Array{Float64, 2}}(itsdone)
        for i in 1:itsdone
            init_sims[i] = rejOutputs[i].init_sims
            init_pars[i] = rejOutputs[i].init_pars
        end
    else
        init_sims = Array{Array{Float64, 2}}(0)
        init_pars = Array{Array{Float64, 2}}(0)
    end
    output = ABCPMCOutput(nparameters, abcinput.nsumstats, itsdone, simsdone, cusims, parameters, sumstats, distances, weights, dists, thresholds, init_sims, init_pars)
end


    
"
Return the perturbation distribution to be used given output `x` from the previous iteration.
Argument `diag` specifies whether the perturbation should have a diagonal covariance.
"
function getperturbdist(x::ABCRejOutput, diag::Bool)
    wv = Weights(x.weights)
    if (diag)
        ##Calculate diagonalised variance of current weighted particle approximation
        diagvar = Float64[var(x.parameters[i,:], wv, corrected=false) for i in 1:x.nparameters]
        perturbdist = MvNormal(2.0 .* diagvar)
    else
        ##Calculate variance of current weighted particle approximation
        currvar = cov(x.parameters, wv, 2, corrected=false)
        perturbdist = MvNormal(2.0 .* currvar)
    end
    return perturbdist
end

##Check if summary statistics meet acceptance requirement
function propgood(s::Array{Float64, 1}, dist::ABCDistance, threshold::Float64)
    ##n.b. threshold==Inf gives acceptance even if dist is not initialised
    threshold == Inf || evaldist(dist, s)<=threshold
end

##Check if summary statistics meet all of previous acceptance requirements
function propgood(s::Array{Float64, 1}, dists::Array{ABCDistance, 1}, thresholds::Array{Float64, 1})
    for i in length(dists):-1:1 ##Check the most stringent case first
        if !propgood(s, dists[i], thresholds[i])
            return false
        end
    end
    return true
end

##Samples from importance density defined by prev output
function rimportance(out::ABCRejOutput, dist::MvNormal)
    i = sample(Weights(out.weights))
    out.parameters[:,i] + rand(dist)
end

"
Calculate importance weight of parameters `x` given `priorweight`, their prior density, `old`, the output of the previous iteration and `perturbdist`, the perturbation distribution used in the importance density.
"
function get1weight(x::Array{Float64,1}, priorweight::Float64, old::ABCRejOutput, perturbdist::MvNormal)
    nparticles = size(old.parameters)[2]
    temp = Float64[pdf(perturbdist, x-old.parameters[:,i]) for i in 1:nparticles]
    priorweight / sum(old.weights .* temp)
end

"
Calculates importance weights for parameters `pars` (columns are parameter vectors) given `priorweights`, their priors densities, `old`, the output of the previous iteration and `perturbdist`, the perturbation distribution used in the importance density.
"
function getweights(pars::Array{Float64, 2}, priorweights::Array{Float64,1}, old::ABCRejOutput, perturbdist::MvNormal)
    nparticles = size(pars)[2]
    weights = Float64[get1weight(pars[:,i], priorweights[i], old, perturbdist) for i in 1:nparticles]
    weights ./ sum(weights)
end
