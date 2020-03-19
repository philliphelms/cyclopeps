from cyclopeps.tools.utils import *
from cyclopeps.tools.peps_tools import PEPS
from cyclopeps.ops.asep import return_op
from cyclopeps.algs.tebd import run_tebd
from cyclopeps.algs.tebd import run_tebd
from sys import argv

# Get input values
Nx = 4#int(argv[1])
Ny = 4#int(argv[2])
D  = 2#int(argv[3])
chi= 10#int(argv[4])
d  = 2

# TEBD Parameters
step_sizes = [0.1,0.05, 0.01]
n_step =     [100, 100,  100]
chi        = [ 10,  20,   10]

# ---------------------------------------------------------
# Hop to the right
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
sx = -0.2
sy = 0.
params = (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy)

# Create the Suzuki trotter decomposed operator
ops = return_op(Nx,Ny,params)

# Run TEBD
Ef,_ = run_tebd(Nx,Ny,d,ops,
                D=D,
                su_step_size=[0.5,0.1,0.05],
                su_n_step=[50,50,50],
                su_conv_tol=1e-5,
                su_chi=20,
                chi=chi,
                n_step=n_step,
                step_size=step_sizes)
