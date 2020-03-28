from cyclopeps.tools.utils import *
import copy
from cyclopeps.tools.peps_tools import PEPS
from cyclopeps.ops.j1j2 import return_op
from cyclopeps.ops.heis import return_op as return_heis
from cyclopeps.algs.nntebd import run_tebd as fu
from cyclopeps.algs.simple_update import run_tebd as su
from sys import argv
import numpy as np

# Get inputs
Nx = int(argv[1])
Ny = int(argv[2])
D  = int(argv[3])
chi= int(argv[4])
backend = 'numpy'
d = 2

# TEBD Parameters
step_sizes = [0.1,0.05, 0.01]
n_step     = [500, 500,  500]
chi        = [ 10,  20,   50]
conv_tol   = [1e-5,1e-5,1e-5]

# Get mpo
ham = return_op(Nx,Ny,sym=None,backend=backend)
heisham = return_heis(Nx,Ny,sym=None,backend=backend)

# Create the peps
peps = PEPS(Nx=Nx,
            Ny=Ny,
            d=d,
            D=D,
            chi=chi[0],
            backend=backend)

# Calculate the operator
E1 = peps.calc_op(ham,chi=chi,nn=True,contracted_env=True)
print(E1)
E2 = peps.calc_op(ham,chi=chi,nn=True,contracted_env=False)
print(E2)
E3 = peps.calc_op(heisham)
print(E3)

import sys
sys.exit()
# Now, do it for two different PEPS






# Run SU/FU
Ef,_ = fu(Nx,
          Ny,
          d,
          ham,
          D=D,
          chi=chi,
          backend=backend,
          n_step=n_step,
          step_size=step_sizes,
          conv_tol=conv_tol)

print('\n\nFinal  E = {}'.format(Ef))
