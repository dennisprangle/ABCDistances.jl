module ABCDistances

using StatsBase.WeightVec, StatsBase.cov, StatsBase.sample, StatsBase.mean, StatsBase.ordinalrank
using Distributions
import Base.show, Base.copy

export
  ABCDistance, Lp, Euclidean, Logdist, MahalanobisDiag, MahalanobisEmp, ##ABCDistance types
  init, evaldist, ##ABCDistance methods
  ABCInput, RefTable, ABCOutput, ABCRejOutput, ABCSMCOutput, ##General ABC types
  sortABCOutput!, show, copy, parameter_means, parameter_vars, parameter_covs, ##General ABC methods
  abcRejection, abcSMC, ##ABC algorithms
  rgk, rgk_os, ##g&k methods
  Stoichiometry, gillespie_partial_sim, gillespie_sim ##Gillespie algorithm
  
include("distances.jl")
include("abc.jl")
include("abc_rejection.jl")
include("abc_smc.jl")
include("gk.jl")
include("gillespie.jl")
end
