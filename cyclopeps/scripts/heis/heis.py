import unittest
from cyclopeps.tools.utils import *
import copy
from cyclopeps.tools.peps_tools import PEPS
from cyclopeps.ops.heis import return_op
from cyclopeps.algs.tebd import run_tebd as fu
from cyclopeps.algs.simple_update import run_tebd as su
from sys import argv

# Get inputs
Ny = 10
Nx = 10
d  = 2
D = 2
chi= 20
step=[0.1,0.01]
nstep=[100,25]


# Get mpo
ham = return_op(Nx,Ny)

# Run SU/FU
Ef,_ = fu(Nx,Ny,d,ham,D=D,chi=chi,n_step=nstep,step_size=step)

print('\n\nFinal  E = {}'.format(Ef))
