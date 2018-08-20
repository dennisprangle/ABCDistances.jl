module ABCDistances

using StatsBase: Weights, cov, sample, mean
using Distributions
using ProgressMeter
import Base.show, Base.copy

export
  ABCDistance, Lp, Euclidean, Logdist, WeightedEuclidean, MahalanobisEmp, ##ABCDistance types
  init, evaldist, ##ABCDistance methods
  ABCInput, RefTable, ABCOutput, ABCRejOutput, ABCPMCOutput, ##General ABC types
  sortABCOutput!, show, copy, parameter_means, parameter_vars, parameter_covs, ##General ABC methods
  abcRejection, abcPMC2, abcPMC3, abcPMC3V, abcPMC4, abcPMC5, ##ABC algorithms
  rgk, rgk_os, ##g&k methods
  Stoichiometry, gillespie_partial_sim, gillespie_sim ##Gillespie algorithm
  
include("distances.jl")
include("abc.jl")
include("abc_rejection.jl")
include("abc_pmc.jl")
include("abc_pmc_interface.jl")
include("gk.jl")
include("gillespie.jl")
end
