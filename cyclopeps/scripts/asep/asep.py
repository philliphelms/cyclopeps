from cyclopeps.tools.utils import *
from cyclopeps.tools.peps_tools import PEPS
from cyclopeps.ops.asep import return_op
from cyclopeps.algs.tebd import run_tebd
from sys import argv

# Get input values
Nx = 4#int(argv[1])
Ny = 4#int(argv[2])
D  = 2#int(argv[3])
chi= 10#int(argv[4])
d  = 2

# TEBD Parameters
step_sizes = [1.0,0.1,0.05, 0.01]
n_step =     [100,100, 100,  100]
chi        = [  2, 10,  20,   10]

# ---------------------------------------------------------
# Hop to the right
# ASEP params
jr = 1.
jl = 0.
ju = 0.
jd = 0.
cr = 0.35
cl = 0.
cu = 0.
cd = 0.
dr = 2./3.
dl = 0.
du = 0.
dd = 0.
sx = 0.2
sy = 0.
params = (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy)

# Create the Suzuki trotter decomposed operator
ops = return_op(Nx,Ny,params)

# Run TEBD
Ef = run_tebd(Nx,Ny,d,ops,D=D,chi=chi,n_step=n_step,step_size=step_sizes)

"""
# ---------------------------------------------------------
# Hop to the left:
# ASEP params
jrn = jl
jln = jr
jun = ju
jdn = jd
crn = cl
cln = cr
cun = cu
cdn = cd
drn = dl
dln = dr
dun = du
ddn = dd
sxn = -sx
syn = sy
params = (jrn,jln,jun,jdn,crn,cln,cun,cdn,drn,dln,dun,ddn,sxn,syn)

# Create the Suzuki trotter decomposed operator
ops = return_op(Nx,Ny,params)

# Run TEBD
Ef = run_tebd(Nx,Ny,d,ops,D=D,chi=chi,n_step=n_step,step_size=step_sizes)

# ---------------------------------------------------------
# Hop upwards:
# ASEP params
jrn = ju
jln = jd
jun = jr
jdn = jl
crn = cu
cln = cd
cun = cr
cdn = cl
drn = du
dln = dd
dun = dr
ddn = dl
sxn = sy
syn = sx
params = (jrn,jln,jun,jdn,crn,cln,cun,cdn,drn,dln,dun,ddn,sxn,syn)

# Create the Suzuki trotter decomposed operator
ops = return_op(Ny,Nx,params)

# Run TEBD
Ef = run_tebd(Ny,Nx,d,ops,D=D,chi=chi,n_step=n_step,step_size=step_sizes)

# ---------------------------------------------------------
# Hop downwards
# ASEP params
jrn = ju
jln = jd
jun = jl
jdn = jr
crn = cu
cln = cd
cun = cl
cdn = cr
drn = du
dln = dd
dun = dl
ddn = dr
sxn = sy
syn = -sx
params = (jrn,jln,jun,jdn,crn,cln,cun,cdn,drn,dln,dun,ddn,sxn,syn)

# Create the Suzuki trotter decomposed operator
ops = return_op(Ny,Nx,params)

# Run TEBD
Ef = run_tebd(Ny,Nx,d,ops,D=D,chi=chi,n_step=n_step,step_size=step_sizes)
"""
