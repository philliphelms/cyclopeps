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
step_size = 0.05
n_step = 500

# Get mpo
if Zn is None:
    ham = return_op(Nx,Ny,sym=None,backend=backend)
else:
    ham = return_op(Nx,Ny,sym='Z2',backend=backend)

# Run SU
"""
Ef,pepssu = su(Nx,
               Ny,
               d,
               ham,
               thermal=True,
               D=D,
               Zn=Zn,
               chi=chi,
               backend=backend,
               n_step=n_step,
               step_size=step_size,
               conv_tol=0.)
"""

# Run SU/FU
Ef,pepsfu = fu(Nx,
               Ny,
               d,
               ham,
               thermal=True,
               D=D,
               Zn=Zn,
               chi=chi,
               backend=backend,
               n_step=n_step,
               step_size=step_size,
               conv_tol=0)

print('\n\nFinal  E = {}'.format(Ef))
