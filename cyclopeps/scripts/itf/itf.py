import unittest
from cyclopeps.tools.utils import *
import copy
from cyclopeps.tools.peps_tools import PEPS
from cyclopeps.ops.itf import return_op
from cyclopeps.algs.tebd import run_tebd as fu
from cyclopeps.algs.simple_update import run_tebd as su
from sys import argv

# Get inputs
Ny = int(argv[1])
Nx = int(argv[2])
d  = int(argv[3])
D  = int(argv[4])
chi= int(argv[5])

# Get mpo
ham = return_op(Nx,Ny,(1.,2.))

# Run SU/FU
Ef,_ = fu(Nx,Ny,d,ham,D=D,chi=chi,n_step=100)

print('\n\nFinal  E = {}'.format(Ef))
