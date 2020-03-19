import unittest
import time
import copy
from cyclopeps.tools.peps_tools import PEPS
from sys import argv
import cProfile
import pstats

# Get inputs
Ny = int(argv[1])
Nx = int(argv[2])
d  = 2
D  = int(argv[3])
chi= int(argv[4])
Zn = int(argv[5])
backend = 'numpy'

# Create a random pep
peps = PEPS(Nx=Nx,
            Ny=Ny,
            d=d,
            D=D,
            chi=chi,
            Zn=Zn,
            backend=backend,
            normalize=False)

# contract the norm
fname = argv[6]+'_norm_stats_Nx'+str(Nx)+'_Ny'+str(Ny)+'_d'+str(d)+'_D'+str(D)+'_chi'+str(chi)
t0 = time.time()
cProfile.run('norm = peps.calc_norm(chi=chi)',fname+'_symtensor')
tf = time.time()
print('Symtensor Contraction Time = {}'.format(tf-t0))
peps_sparse = peps.make_sparse()
t0 = time.time()
cProfile.run('norm1 = peps_sparse.calc_norm(chi=chi)',fname+'_slow')
print('Slow Contraction Time = {}'.format(tf-t0))
