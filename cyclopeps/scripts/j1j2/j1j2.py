from cyclopeps.tools.utils import *
from cyclopeps.tools.peps_tools import PEPS
from cyclopeps.ops.j1j2 import return_op
from cyclopeps.ops.heis import return_op as return_heis
from cyclopeps.algs.nntebd import run_tebd
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
step_sizes = [0.1,0.01,0.001]
n_step     = 100
conv_tol   = 1e-10

# Get mpo
ham = return_op(Nx,Ny,sym=None,backend=backend)

# Run SU/FU
Ef,_ = run_tebd(Nx,
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
