# cyclomps
An implementation of common algorithms for Projected Entangled Pair States (PEPS)
using Cyclops Tensor Framework (CTF).

# Dependencies
Currently, the only dependency is the python version of 
CTF.

## To Do List
* Function to increase bond dimension of PEPS
* Profile CTF Calculations to see how they scale with number of processors

## Known Bugs
* TEBD converges to nearly the correct energies for ITF model

### CTF Problems/Questions
* Cannot use -1 as and index
* 1D CTF arrays do not have the attribute __len__
