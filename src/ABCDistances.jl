module ABCDistances

using  StatsBase.WeightVec, StatsBase.cov, StatsBase.sample, StatsBase.mean
import Base.show, Base.copy
import Distributions.Normal, Distributions.MvNormal, Distributions.Gamma, Distributions.rand, Distributions.pdf, Distributions.Exponential, Distributions.Categorical

export
  ABCNorm, Lp, Euclidean, Lognorm, MahalanobisDiag, MahalanobisEmp, ##ABCNorm types
  ##Also export MahalanobisGlasso when finished
  init, evalnorm, ##ABCNorm methods
  ABCInput, RefTable, ABCOutput, ABCRejOutput, ##General ABC types
  sortABCOutput!, show, copy, parameter_means, parameter_vars, parameter_covs, ##General ABC methods
  abcRejection, abcSMC, ##ABC algorithms
  rgk, rgk_os, ##g&k methods
  Stoichiometry, gillespie_partial_sim, gillespie_sim ##Gillespie algorithm
  
include("norms.jl")
include("abc.jl")
include("abc_rejection.jl")
include("abc_smc.jl")
include("gk.jl")
include("gillespie.jl")
end
