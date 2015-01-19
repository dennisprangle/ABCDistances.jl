module ABCNorms

using  StatsBase.WeightVec, StatsBase.cov, StatsBase.sample, StatsBase.mean
import Base.show
import Distributions.MvNormal, Distributions.rand, Distributions.pdf

export
  ABCNorm,
  Lp,
  Euclidean,
  Lognorm,
  MahalanobisDiag,
  MahalanobisEmp,
  init,
  evalnorm,
  ABCInput,
  RefTable,
  ABCOutput,
  sortABCOutput!,
  rgk,
  show,
  doABCSMC
  ##Also export MahalanobisGlasso when finished
  
include("norms.jl")
include("abc.jl")
include("abc_rejection.jl")
include("abc_smc.jl")
include("gk.jl")
end
