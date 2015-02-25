##Simulate forward 1 event
##Using stoichiometry matrix "S", current state "x" and constants "θ"
##Return tuple of time till next event (can be Inf) and event number
function gillespie_step(S::Array{Unsigned, 2}, x::Array{Unsigned, 1}, θ::Array{FloatingPoint, 1})
    ##rates = ???
    ##total_rate = sum(rates)
    ##Δt = total_rate * ??? (draw from an exponential dist)
    ##event = ??? (draw from a categorical dist) Or select arbitrary value if Δt==Inf
    ##(Δt, event)
end

##Perform a simulation of the Gillespie algorithm
##with stoichiometry matrix "S", initial state "x0" and constants "θ"
##A matrix is returned with entry [:,i] the state at time obs_times[i]
returning state at specified timepoints
function gillespie_sim(S::Array{Unsigned, 2}, x0::Array{Unsigned, 1}, θ::Array{FloatingPoint, 1}, obs_times::Array{FloatingPoint, 1})
    ##Initialise output matrix
    ##Record initial state if obs_times[1]=0
    ##Work out next observation time
    ##Set current time to zero
    ##MAIN LOOP
        ##Perform a step
        ##If next event time is after next observation then...
           ##Record current state in output matrix
           ##Update current time to observation time
        ##Else
           ##Update state
           ##Update current time to next event time
        ##
        ##Quit if things have gone haywire??
    ##END LOOP
end


