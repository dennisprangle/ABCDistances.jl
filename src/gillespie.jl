##Details of reactions in a convenient format for later use
immutable Stoichiometry
    nspecies::Int32 ##How many species in total
    nreactions::Int32 ##How many reactions in total
    input_length::Int32 ##Length of the input arrays below
    ##Reaction input_reaction[i] requires input_index[i] units of species input_species[i]
    input_reactions::Array{Int32, 1}
    input_species::Array{Int32, 1}
    input_indices::Array{Int32, 1}
    ##change[i,j] represents net change in number of units of species i from reaction j
    change::Array{Int32, 2}

    function Stoichiometry(nspecies, nreactions, input_length, input_reactions, input_species, input_indices, change)
        if (nspecies < 1)
            error("Must have at least 1 species")
        elseif (nreactions < 1)
            error("Must have at least 1 reaction")
        elseif (length(input_reactions) != input_length)
            error("input_length inconsistent with input_reactions length")
        elseif (length(input_species) != input_length)
            error("input_length inconsistent with input_species length")
        elseif (length(input_indices) != input_length)
            error("input_length inconsistent with input_indices length")
        elseif (maximum(input_reactions) > nreactions)
            error("input_reactions values must not be greater than nreactions")
        elseif (minimum(input_reactions) <= 0)
            error("input_reactions values must be positive")
        elseif (maximum(input_species) > nspecies)
            error("input_species values must not be greater than nspecies")
        elseif (minimum(input_species) <= 0)
            error("input_species values must be positive")
        elseif (minimum(input_indices) <= 0)
            error("input_indices values must be positive")
        elseif (size(change) != (nspecies, nreactions))
            error("change matrix dimensions incorrect")
        end            
        new(nspecies, nreactions, input_length, input_reactions, input_species, input_indices, change)
    end
end

##Stoichiometry constructor using P (inputs) and Q (outputs) matrices as in Owen et at 2014. (i.e. P[i,j] is number of species j required for reaction i, and Q[i,j] is number of species j output by reaction i).
function Stoichiometry(P::Array{Int32, 2}, Q::Array{Int32, 2})
    if (size(P) != size(Q))
        error("Dimensions of P and Q must be the same")
    elseif (minimum(P) < 0)
        error("All P elements must be non-negative")
    elseif (minimum(Q) < 0)
        error("All Q elements must be non-negative")
    end
    (nreactions, nspecies) = size(P)
    input_length = 0
    input_reactions = Array(Int32, 0)
    input_species = Array(Int32, 0)
    input_indices = Array(Int32, 0)
    ##Store the non-zero elements of P
    for (this_reaction in 1:nreactions)
        for (this_species in 1:nspecies)
            this_index = P[this_reaction, this_species]
            if (this_index > 0)
                push!(input_reactions, this_reaction)
                push!(input_species, this_species)
                push!(input_indices, this_index)
                input_length += 1
            end
        end
    end
    Stoichiometry(nspecies, nreactions, input_length, input_reactions, input_species, input_indices, (Q-P)')
end

##Simulate forward 1 event
##Using stoichiometry "s", current state "state" and constants "θ"
##Return tuple of time till next event (can be Inf) and event number
function gillespie_step(s::Stoichiometry, state::Array{Int32, 1}, θ::Array{Float64, 1})
    rates = copy(θ)
    expdist = Exponential()  
    for (i in 1:s.input_length)
        rates[s.input_reactions[i]] *= state[s.input_species[i]]^s.input_indices[i]
    end
    total_rate = sum(rates)
    Δt = rand(expdist) / total_rate
    if (total_rate == 0.0)
        event = 1 ##An arbitrary value as event type irrelevant
    else
        event = rand(Categorical(rates/total_rate))
    end
    (Δt, event)
end

##Perform a simulation of the Gillespie algorithm
##with stoichiometry "s", initial state "state0" and constants "θ"
##A matrix is returned with entry [:,i] the state at time obs_times[i] returning state at specified timepoints
function gillespie_partial_sim(s::Stoichiometry, state0::Array{Int32, 1}, θ::Array{Float64, 1}, obs_times::Array{Float64, 1})
    if (length(state0) != s.nspecies)
        error("Number of species given in initial state and stoichiometry do not match")
    elseif (length(θ) != s.nreactions)
        error("Number of reactions given in θ and stoichiometry do not match")
    elseif (minimum(θ) < 0)
        error("Negative values in θ")
    elseif (!issorted(obs_times, lt=((x,y)->x<=y)))
        error("obs_times must be strictly increasing")
    elseif (obs_times[1] < 0)
        error("obs_times must be non-negative")
    end
    nobs = length(obs_times)
    observations = Array(Int32, (s.nspecies, nobs))
    obs_count = 1 ##Which observation we are currently looking for
    if (obs_times[1] == 0)
        observations[:,1] = state0
        obs_count += 1
    end
    t_curr = 0.0
    state_curr = copy(state0)
    t_next = obs_times[obs_count]    
    while (true)
        (Δt, event) = gillespie_step(s, state_curr, θ)
        if (Δt > t_next - t_curr)
            observations[:,obs_count] = state_curr
            t_curr = t_next
            obs_count += 1
            if (obs_count > nobs)
                break
            else
                t_next = obs_times[obs_count]
            end
        else
            state_curr += s.change[:,event]
            t_curr += Δt
        end
    end
    observations
end

##Perform a simulation of the Gillespie algorithm
##with stoichiometry "s", initial state "state0" and constants "θ"
##Until (a) time maxt is reached (b) maxit iterations have been performed (c) No further events possible.
##The output is a vector of event times and a matrix whose columns are corresponding states
function gillespie_sim(s::Stoichiometry, state0::Array{Int32, 1}, θ::Array{Float64, 1}, maxt::FloatingPoint, maxit::Integer)
    if (length(state0) != s.nspecies)
        error("Number of species given in initial state and stoichiometry do not match")
    elseif (length(θ) != s.nreactions)
        error("Number of reactions given in θ and stoichiometry do not match")
    elseif (minimum(θ) < 0)
        error("Negative values in θ")
    end
    event_times = [0.0]
    observations = Array(Int32, (s.nspecies, 1))
    observations[:,1] = state0
    obs_count = 2 ##Which observation we are currently looking for
    t_curr = 0.0
    state_curr = copy(state0)
    while (obs_count <= maxit)
        (Δt, event) = gillespie_step(s, state_curr, θ)
        t_curr += Δt
        if (t_curr == Inf || t_curr > maxt)
            break
        end
        state_curr += s.change[:,event]
        push!(event_times, t_curr)
        observations = hcat(observations, copy(state_curr))
        obs_count += 1
    end
    (event_times, observations)
end


