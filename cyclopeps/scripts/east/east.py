from cyclopeps.tools.utils import *
from cyclopeps.tools.peps_tools import PEPS
from cyclopeps.ops.east import return_op
from cyclopeps.algs.tebd import run_tebd
from sys import argv
from numpy import logspace,linspace

# Get input values
Nx = int(argv[1])
Ny = int(argv[2])
D  = int(argv[3])
chi= int(argv[4])
d  = 2
c = 0.2

# Bias sweeps
sVec = linspace(-0.5,0.5,30)

# TEBD Parameters
step_sizes = [0.1,0.05, 0.01]
n_step =     [ 50,  50,   50]
chi        = [ 10,  20,   50]

# ---------------------------------------------------------
E = []
peps = None
for i,s in enumerate(sVec):
    # Create the Suzuki trotter decomposed operator
    params = (c,s)
    ops = return_op(Nx,Ny,params,hermitian=False)
    # Run TEBD
    Ef,peps = run_tebd(Nx,
                       Ny,
                       d,
                       ops,
                       peps=peps,
                       D=D,
                       chi=chi,
                       n_step=n_step,
                       step_size=step_sizes)

    E.append(Ef)

    for s2 in range(len(E)):
        print('{}\t{}'.format(sVec[s2],E[s2]))

