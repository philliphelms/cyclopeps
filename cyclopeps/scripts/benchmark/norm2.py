from cyclopeps.tools.utils import *
import copy
from cyclopeps.tools.peps_tools import PEPS
import time
from sys import argv


Nx  = int(argv[1])
Ny  = int(argv[2])
d   = 2
D   = int(argv[3])
chi = int(argv[4])
Zn  = 2 # Zn symmetry (here, Z2)
backend  = 'numpy'
# Generate random PEPS
peps = PEPS(Nx=Nx,
            Ny=Ny,
            d=d,
            D=D,
            chi=chi,
            Zn=Zn,
            backend=backend,
            normalize=False)
# Compute the norm (2 ways for comparison)
t0 = time.time()
norm0 = peps.calc_norm(chi=chi) 
tf = time.time()
print('Dense contraction time = {}'.format(tf-t0))
peps_sparse = peps.make_sparse()
t0 = time.time()
norm1 = peps_sparse.calc_norm(chi=chi)
tf = time.time()
print('Sparse contraction time = {}'.format(tf-t0))
