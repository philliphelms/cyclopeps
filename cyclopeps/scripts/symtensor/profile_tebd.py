import unittest
from cyclopeps.tools.utils import *
import copy
from cyclopeps.tools.peps_tools import PEPS
from cyclopeps.ops.heis import return_op
from cyclopeps.algs.tebd import tebd_step
from cyclopeps.algs.simple_update import run_tebd as su
from sys import argv
import cProfile
from memory_profiler import profile 

# Get inputs
Nx = int(argv[1])
Ny = int(argv[2])
D  = int(argv[3])
chi= int(argv[4])
Zn = int(argv[5])
if Zn == 0:
    Zn = None
backend = argv[6]
N_node = argv[7]
in_mem = True
d = 2

if backend == 'ctf':
    import ctf
    wrld = ctf.comm()

# TEBD Parameters
step_sizes = [0.5]
n_step = [5]

tmpprint('Starting Calculation')
# Get Hamiltonian
if Zn is None:
    ham  = return_op(Nx,Ny,sym=None,backend=backend)
else:
    ham = return_op(Nx,Ny,sym='Z2',backend=backend)
tmpprint('Got Hamiltonian Ops')

# Create PEPS 
peps = PEPS(Nx,Ny,d,D,chi,Zn=Zn,chi_norm=10,chi_op=10,backend=backend,normalize=False,in_mem=in_mem)
tmpprint('Got PEPS')

# Setup Profiling
profile_fname = 'run_tebd_stats_denseals_Nx{}_Ny{}_d{}_D{}_chi{}_Zn{}_{}_inmem_{}_Nnode{}'.format(Nx,Ny,d,D,chi,Zn,backend,in_mem,N_node)
if backend == 'ctf': 
    from ctf import timer_epoch
    te = timer_epoch('1')
    te.begin()
t0 = time.time()
# Evaluate Operator
cProfile.run('out = tebd_step(peps,ham,0.1,D,chi=chi,als_iter=20,als_tol=1e-6,in_mem={})'.format(in_mem),profile_fname)
tf = time.time()
print('Total time = {}'.format(tf-t0))
# Print Results
if backend == 'ctf': 
    te.end()
