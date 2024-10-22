from cyclopeps.tools.utils import *
from cyclopeps.tools.peps_tools import PEPS
from cyclopeps.ops.j1j2 import return_op
from cyclopeps.algs.nntebd import run_tebd
from sys import argv
import numpy as np

# Get inputs
Nx = int(argv[1])
Ny = int(argv[2])
D  = int(argv[3])
chi= int(argv[4])
dtind = int(argv[5])
j2ind = int(argv[6])
backend = 'numpy'
d = 2

# TEBD Parameters
step_size = [0.1,0.01][dtind]
n_step     = int(10./step_size)

# Get mpo
j2 = [0.1,0.5,1.0][j2ind]
ham = return_op(Nx,Ny,j1=1.,j2=j2,sym=None,backend=backend)

# Run SU/FU
Ef,_ = run_tebd(Nx,
                Ny,
                d,
                ham,
                thermal=True,
                D=D,
                chi=chi,
                backend=backend,
                n_step=n_step,
                full_update=False,
                step_size=step_size)

print('\n\nFinal  E = {}'.format(Ef))
