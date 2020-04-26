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
peps_fdir = '/central/groups/changroup/members/phelms/fu_sweep/'
peps_fname = '{}x{}_D{}_chi{}_dt{}_j2{}'.format(Nx,Ny,D,chi,dtind,j2ind)

# TEBD Parameters
step_size = [0.1,0.01][dtind]
n_step     = int(5./step_size)
truncate_loc = True
local_chi = chi

# Get mpo
j2 = [0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0][j2ind]
ham = return_op(Nx,Ny,j1=1.,j2=j2,sym=None,backend=backend)

# Run SU/FU
Ef,_ = run_tebd(Nx,
                Ny,
                d,
                ham,
                thermal=True,
                D=D,
                chi=chi,
                chiloc=local_chi,
                backend=backend,
                n_step=n_step,
                als_iter=10,
                als_tol=1e-5,
                truncate_loc=truncate_loc,
                step_size=step_size,
                peps_fdir=peps_fdir,
                peps_fname=peps_fname)

print('\n\nFinal  E = {}'.format(Ef))
