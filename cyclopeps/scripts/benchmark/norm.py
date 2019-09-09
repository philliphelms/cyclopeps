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
d  = int(argv[3])
D  = int(argv[4])
chi= int(argv[5])

# Create a random pep
peps = PEPS(Nx=Nx,Ny=Ny,d=d,D=D,chi=chi,singleLayer=True,normalize=False)

# contract the norm
fname = argv[6]+'_norm_stats_Nx'+str(Nx)+'_Ny'+str(Ny)+'_d'+str(d)+'_D'+str(D)+'_chi'+str(chi)
t0 = time.time()
cProfile.run('norm = peps.calc_norm()',fname)
tf = time.time()

# Print results
print('Norm calc time = {}'.format(tf-t0))
