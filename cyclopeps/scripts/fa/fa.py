from cyclopeps.tools.utils import *
from cyclopeps.tools.peps_tools import PEPS
from cyclopeps.ops.fa import return_op
from cyclopeps.algs.tebd import run_tebd
from sys import argv
from numpy import logspace,linspace

# Get input values
Nx = 3
Ny = 3
D  = 4
chi= 10
d  = 2
c = 0.2
sVec = linspace(-1,1,30)

# TEBD Parameters
step_sizes = [ 0.5, 0.2, 0.1, 0.05, 0.01, 0.001,
               0.5, 0.2, 0.1, 0.05, 0.01, 0.001,
               0.5, 0.2, 0.1, 0.05, 0.01, 0.001,
               0.5, 0.2, 0.1, 0.05, 0.01, 0.001,
               0.5, 0.2, 0.1, 0.05, 0.01, 0.001]
n_step =     [  20,  20,  20,   20,   20,    20,
                20,  20,  20,   20,   20,    20,
                20,  20,  20,   20,   20,    20,
                20,  20,  20,   20,   20,    20,
                20,  20,  20,   20,   20,    20]
conv_tol =   [1e-3,1e-4,1e-5, 1e-5, 1e-5,  1e-5,
              1e-3,1e-4,1e-5, 1e-6, 1e-6,  1e-6,
              1e-3,1e-4,1e-5, 1e-6, 1e-7,  1e-7,
              1e-3,1e-4,1e-6, 1e-6, 1e-7,  1e-8,
              1e-3,1e-4,1e-6, 1e-6, 1e-7,  1e-8]
D          = [   2,   2,   2,    2,    2,     2,
                 4,   4,   4,    4,    4,     4,
                 6,   6,   6,    6,    6,     6,
                 8,   8,   8,    8,    8,     8,
                10,  10,  10,   10,   10,    10]

# ---------------------------------------------------------
E = []
peps = None
for i,s in enumerate(sVec):
    # Create the Suzuki trotter decomposed operator
    params = (c,s)
    ops = return_op(Nx,Ny,params,hermitian=False)
    print('s = {}'.format(s))

    # Run TEBD
    Ef,peps = run_tebd(Nx,
                      Ny,
                      d,
                      ops,
                      peps=None,
                      D=D,
                      chi=chi,
                      n_step=n_step,
                      conv_tol=conv_tol,
                      step_size=step_sizes)

    E.append(Ef)
    for s2 in range(len(E)):
        print('{}\t{}'.format(sVec[s2],E[s2]))

