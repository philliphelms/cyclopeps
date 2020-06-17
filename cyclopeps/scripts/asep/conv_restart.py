from cyclopeps.tools.utils import *
from cyclopeps.tools.peps_tools import PEPS,load_peps
from cyclopeps.ops.asep import return_op
from cyclopeps.algs.tebd import run_tebd
from cyclopeps.algs.tebd import run_tebd
from sys import argv
import numpy as np

# Get input values
fname = argv[1]
Nx = int(argv[2])
Ny = int(argv[3])
start = 3

# ---------------------------------------------------------
# Create an initial peps
peps = PEPS(Nx=Nx,Ny=Ny,fname=fname+'_restart',fdir='./',normalize=False)
peps.load_tensors(fname)

# TEBD Parameters
step_sizes = [0.5,0.1,0.01,0.1,0.01,0.1,0.01,0.1,0.01,0.1,0.01,0.1,0.01,0.1,0.01,0.1,0.01,0.1,0.01,0.1,0.01,0.1,0.01,0.1,0.01,0.1,0.01,0.1,0.01,0.1,0.01,0.1,0.01,0.1,0.01,0.1,0.01,0.1,0.01,0.1,0.01,0.1,0.01,0.1,0.01,0.1,0.01,0.1,0.01,0.1,0.01,0.1,0.01,0.1,0.01,0.1,0.01]
D          = [  1,  1,   1,  2,   2,  2,   2,  2,   2,  3,   3,  3,   3,  3,   3,  4,   4,  4,   4,  4,   4,  5,   5,  5,   5,  5,   5,  6,   6,  6,   6,  6,   6,  7,   7,  7,   7,  7,   7,  8,   8,  8,   8,  8,   8,  9,   9,  9,   9,  9,   9, 10,  10, 10,  10, 10,  10]
chi        = [  1,  1,   1, 20,  20, 40,  40, 60,  60, 20,  20, 40,  40, 60,  60, 20,  20, 40,  40, 60,  60, 20,  20, 40,  40, 60,  60, 20,  20, 40,  40, 60,  60, 20,  20, 40,  40, 60,  60, 20,  20, 40,  40, 60,  60, 20,  20, 40,  40, 60,  60, 20,  20, 40,  40, 60,  60]
n_step =     [500]*len(step_sizes)
d  = 2


# ---------------------------------------------------------
# Create Hamiltonian
# ASEP params
jr = 0.9
jl = 0.1
ju = 0.5
jd = 0.5
cr = 0.5
cl = 0.5
cu = 0.5
cd = 0.5
dr = 0.5
dl = 0.5
du = 0.5
dd = 0.5
sx = -0.5
sy = 0.
params = (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy)

# Create the Suzuki trotter decomposed operator
ops = return_op(peps.Nx,peps.Ny,params)



# Run TEBD
Ef,_ = run_tebd(peps.Nx,peps.Ny,peps.d,ops,
                peps=peps,
                D=D[start:],
                chi=chi[start:],
                n_step=n_step[start:],
                conv_tol=1e-8,
                step_size=step_sizes[start:])
