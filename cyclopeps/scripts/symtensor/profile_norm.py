import unittest
from cyclopeps.tools.utils import *
import copy
from cyclopeps.tools.peps_tools import PEPS
from cyclopeps.ops.heis import return_op
from cyclopeps.algs.tebd import run_tebd as fu
from cyclopeps.algs.simple_update import run_tebd as su
from sys import argv
import cProfile

# Get inputs
Nx = int(argv[1])
Ny = int(argv[2])
D  = int(argv[3])
chi= int(argv[4])
Zn = int(argv[5])
if Zn == 0:
    Zn = None
backend = argv[6]
d = 2

if backend == 'ctf':
    import ctf
    wrld = ctf.comm()

# TEBD Parameters
step_sizes = [0.1]
n_step = [5]

# Get Hamiltonian
if Zn is None:
    ham  = return_op(Nx,Ny,sym=None,backend=backend)
else:
    ham = return_op(Nx,Ny,sym='Z2',backend=backend)

# Create PEPS 
peps = PEPS(Nx,Ny,d,D,chi,Zn=Zn,chi_norm=10,chi_op=10,backend=backend,normalize=False)

# Setup Profiling
profile_fname = 'calc_norm_stats_Nx{}_Ny{}_d{}_D{}_chi{}_Zn{}_{}'.format(Nx,Ny,d,D,chi,Zn,backend)
if backend == 'ctf': 
    from ctf import timer_epoch
    te = timer_epoch('1')
    te.begin()
t0 = time.time()
# Evaluate Operator
cProfile.run('val = peps.calc_norm(chi=chi)',profile_fname)
tf = time.time()
# Print Results
if backend == 'ctf': te.end()
print(tf-t0)
