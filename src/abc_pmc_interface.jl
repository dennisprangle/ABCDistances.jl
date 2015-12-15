############################################################
##Interface to the ABC-PMC algorithms. Mainly documentation.
############################################################

"
Perform a version of ABC-PMC: Algorithm 2 of the paper which is essentially the original Toni et al approach.
Its arguments are:

* `abcinput`
An `ABCInput` variable.

* `N`
Number of accepted particles in each iteration.

* `α`
A tuning parameter between 0 and 1 determining how fast the acceptance threshold is reduced. (In more detail, the acceptance threshold in iteration t is the α quantile of distances from particles which meet the acceptance criteria of the previous iteration.)

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
If `initialise_dist` is true, then this must be left at its default value.

The output is a `ABCPMCOutput` object.
"
function abcPMC2(abcinput::ABCInput, N::Integer, α::Float64, maxsims::Integer, nsims_for_init=10000; store_init=false, diag_perturb=false, silent=false, h1=Inf)
    abcPMC_23(abcinput, N, α, maxsims, false, nsims_for_init, store_init=store_init, diag_perturb=diag_perturb, silent=silent, h1=h1)
end

"
Perform a version of ABC-PMC: Algorithm 3 of the paper which chooses the distance function in the initial iteration.
Its arguments are:

* `abcinput`
An `ABCInput` variable.

* `N`
Number of accepted particles in each iteration.

* `α`
A tuning parameter between 0 and 1 determining how fast the acceptance threshold is reduced. (In more detail, the acceptance threshold in iteration t is the α quantile of distances from particles which meet the acceptance criteria of the previous iteration.)

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
If `initialise_dist` is true, then this must be left at its default value.

The output is a `ABCPMCOutput` object.
"
function abcPMC3(abcinput::ABCInput, N::Integer, α::Float64, maxsims::Integer, nsims_for_init=10000; store_init=false, diag_perturb=false, silent=false, h1=Inf)
    abcPMC_23(abcinput, N, α, maxsims, true, nsims_for_init, store_init=store_init, diag_perturb=diag_perturb, silent=silent, h1=h1)
end
    
"
Perform a version of ABC-PMC.
This is the variant of Algorithm 3 mentioned in Section 4.2 of the paper.
As for Algorithm 3, the distance function is chosen in the initial iteration.
The variation is that the acceptance threshold for iteration t is chosen during that iteration.
Its arguments are as follows:

* `abcinput`
An `ABCInput` variable.

* `N`
Number of accepted particles in each iteration.

* `α`
A tuning parameter between 0 and 1 determining how fast the acceptance threshold is reduced. (In more detail, the acceptance threshold in iteration t is the α quantile of distances from particles which meet the acceptance criteria of the previous iteration.)

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

The output is a `ABCPMCOutput` object.
"
function abcPMC3V(abcinput::ABCInput, N::Integer, α::Float64, maxsims::Integer, nsims_for_init=10000; store_init=false, diag_perturb=false, silent=false)
    abcPMC_3V5(abcinput, N, α, maxsims, false, nsims_for_init, store_init=store_init, diag_perturb=diag_perturb, silent=silent)
end

"
Perform a version of ABC-PMC proposed in the paper, Algorithm 5.
This chooses the distance function for iteration t during that iteration.
Its arguments are as follows:

* `abcinput`
An `ABCInput` variable.

* `N`
Number of accepted particles in each iteration.

* `α`
A tuning parameter between 0 and 1 determining how fast the acceptance threshold is reduced. (In more detail, the acceptance threshold in iteration t is the α quantile of distances from particles which meet the acceptance criteria of the previous iteration.)

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

The output is a `ABCPMCOutput` object.
"
function abcPMC5(abcinput::ABCInput, N::Integer, α::Float64, maxsims::Integer, nsims_for_init=10000; store_init=false, diag_perturb=false, silent=false)
    abcPMC_3V5(abcinput, N, α, maxsims, true, nsims_for_init, store_init=store_init, diag_perturb=diag_perturb, silent=silent)
end
