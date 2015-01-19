module ABCNorms

using  StatsBase.WeightVec, StatsBase.cov, StatsBase.sample, StatsBase.mean
import Base.show, Base.copy
import Distributions.MvNormal, Distributions.rand, Distributions.pdf

export
  ABCNorm, Lp, Euclidean, Lognorm, MahalanobisDiag, MahalanobisEmp, ##ABCNorm types
  ##Also export MahalanobisGlasso when finished
  init, evalnorm, ##ABCNorm methods
  ABCInput, RefTable, ABCOutput, ##General ABC types
  sortABCOutput!, show, copy, ##General ABC methods
  rgk, ##g&k methods
  doABCSMC ##ABCSMC methods
  
include("norms.jl")
include("abc.jl")
include("abc_rejection.jl")
include("abc_smc.jl")
include("gk.jl")
end
