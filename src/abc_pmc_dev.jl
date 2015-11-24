"
Perform a version of ABC-PMC.
This implements algorithm ? of the paper.
Arguments are as for `abcPMC` with one removal (`adaptive`) and two additions

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

* `adaptive` (optional)
Whether the distance function should be adapted at each iteration.
By default this is false, giving the variant algorithm mentioned in Section 4.1 of the paper.
When true Algorithm 4 of the paper is implemented.

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
function abcPMC_dev(abcinput::ABCInput, N::Integer, α::Float64, maxsims::Integer, nsims_for_init=10000; store_init=false, diag_perturb=false, silent=false, h1=Inf)
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
        newparameters = Array(Float64, (nparameters, N))
        newsumstats = Array(Float64, (abcinput.nsumstats, N))
        newpriorweights = Array(Float64, N)
        successes_thisit = 0            
        nextparticle = 1
        ##Initialise storage of simulated parameter/summary pairs
        sumstats_forinit = Array(Float64, (abcinput.nsumstats, nsims_for_init))
        pars_forinit = Array(Float64, (nparameters, nsims_for_init))
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
    parameters = Array(Float64, (nparameters, N, itsdone))
    sumstats = Array(Float64, (abcinput.nsumstats, N, itsdone))
    distances = Array(Float64, (N, itsdone))
    weights = Array(Float64, (N, itsdone))
    for i in 1:itsdone        
        parameters[:,:,i] = rejOutputs[i].parameters
        sumstats[:,:,i] = rejOutputs[i].sumstats
        distances[:,i] = rejOutputs[i].distances
        weights[:,i] = rejOutputs[i].weights
    end
    if (store_init)
        init_sims = Array(Array{Float64, 2}, itsdone)
        init_pars = Array(Array{Float64, 2}, itsdone)
        for i in 1:itsdone
            init_sims[i] = rejOutputs[i].init_sims
            init_pars[i] = rejOutputs[i].init_pars
        end
    else
        init_sims = Array(Array{Float64, 2}, 0)
        init_pars = Array(Array{Float64, 2}, 0)
    end
    output = ABCPMCOutput(nparameters, abcinput.nsumstats, itsdone, simsdone, cusims, parameters, sumstats, distances, weights, dists[1:itsdone], thresholds[1:itsdone], init_sims, init_pars)
end
