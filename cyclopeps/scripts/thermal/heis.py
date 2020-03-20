import unittest
from cyclopeps.tools.utils import *
import copy
from cyclopeps.tools.peps_tools import PEPS
from cyclopeps.ops.heis import return_op
from cyclopeps.algs.tebd import run_tebd as fu
from cyclopeps.algs.simple_update import run_tebd as su
from sys import argv

# Get inputs
Nx = int(argv[1])
Ny = int(argv[2])
D  = int(argv[3])
chi= int(argv[4])
Zn = int(argv[5])
if Zn == 0:
    Zn = None
backend = 'numpy'
d = 2

# TEBD Parameters
step_sizes = [0.1,0.05, 0.01]
n_step     = [500, 500,  500]
chi        = [ 10,  20,   50]
conv_tol   = [1e-5,1e-5,1e-5]

# Get mpo
if Zn is None:
    ham = return_op(Nx,Ny,sym=None,backend=backend)
else:
    ham = return_op(Nx,Ny,sym='Z2',backend=backend)

# Run SU/FU
Ef,_ = fu(Nx,
          Ny,
          d,
          ham,
          thermal=True,
          D=D,
          Zn=Zn,
          chi=chi,
          backend=backend,
          n_step=n_step,
          step_size=step_sizes,
          conv_tol=conv_tol,
          su_step_size=step_sizes,
          su_chi=chi[0],
          su_conv_tol=conv_tol,
          su_n_step=n_step)

print('\n\nFinal  E = {}'.format(Ef))
