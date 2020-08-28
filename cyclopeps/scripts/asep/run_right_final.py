from cyclopeps.tools.utils import *
from cyclopeps.tools.peps_tools import PEPS
from cyclopeps.tools.ops_tools import ops_conj_trans
from cyclopeps.ops.asep import return_op,return_curr_op
from cyclopeps.ops.basic import return_dens_op
from cyclopeps.algs.tebd import run_tebd
from sys import argv
from numpy import linspace


# Input arguments
Nx = int(argv[1])
Ny = int(argv[2])
sxind = int(argv[3])
syind = int(argv[4])
optind= int(argv[5])

# Calculation parameters
dt     = [0.5,0.1,0.05,0.01,0.001,0.0001]
D   = [1]+[2]*6+[3]*6+[4]*6+[5]*6
chi = [1]+[20,40,60,80,100,150]*4
D = D[optind]
chi = chi[optind]
D      = [D]*len(dt)
chi    = [chi]*len(dt)
conv   = [1e-5,1e-6,1e-7,1e-8,1e-10,1e-10]
n_step = [100,200,300,500,1000,1000]
d = 2

# Sx parameters
sxVec = linspace(-0.5,0.5,21)
syVec = linspace(-0.5,0.5,21)

# Filenames for saved PEPS
savedir = "/central/groups/changroup/members/phelms/asep/final/peps/"
prepend = argv[6]
fnamer = prepend+"Nx{}_Ny{}_sx{}_sy{}_run_right".format(Nx,Ny,sxind,syind)

# ---------------------------------------------------------
# Hop to the right
# ASEP params
jr = 0.9
jl = 1.-jr
ju = 0.9
jd = 1.-ju
cr = 0.5
cl = 0.5
cu = 0.5
cd = 0.5
dr = 0.5
dl = 0.5
du = 0.5
dd = 0.5
sx = sxVec[sxind]
sy = syVec[syind]
params = (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy)
print('params:\n')
print('Nx = {}'.format(Nx))
print('Ny = {}'.format(Ny))
print('jr = {}'.format(jr))
print('jl = {}'.format(jl))
print('ju = {}'.format(ju))
print('jd = {}'.format(jd))
print('cr = {}'.format(cr))
print('cl = {}'.format(cl))
print('cu = {}'.format(cu))
print('cd = {}'.format(cd))
print('dr = {}'.format(dr))
print('dl = {}'.format(dl))
print('du = {}'.format(du))
print('dd = {}'.format(dd))
print('sx = {}'.format(sx))
print('sy = {}'.format(sy))

# Create the Suzuki trotter decomposed operator
ops = return_op(Nx,Ny,params)

# Run TEBD
peps = PEPS(Nx,Ny,d,D[0],chi[0],
            fname=fnamer,
            fdir=savedir,
            norm_tol=0.5,
            norm_bs_upper=3.,
            norm_bs_lower=0.)

# Loop over all optimizaton parameters
for ind in range(len(D)):
    # Update PEPS Parameters
    peps.D = D[ind]
    peps.chi = chi[ind]
    peps.fname = prepend+"Nx{}_Ny{}_sx{}_sy{}_D{}_chi{}_run_right".format(Nx,Ny,sxind,syind,D[ind],chi[ind])

    # --------------------------------------------------------------------
    # Calculate right eigenstate
    for i in range(5):
        try:
            Ef,peps = run_tebd(Nx,
                               Ny,
                               d,
                               ops,
                               peps=peps,
                               D=D[ind],
                               chi=chi[ind],
                               n_step=n_step[ind],
                               step_size=dt[ind],
                               conv_tol=conv[ind])
            break
        except Exception as e:
            print('Failed Right TEBD:\n{}'.format(e))
            print('Restarting ({}/{} restarts)'.format(i,5))
            peps.load_tensors(peps.fdir+peps.fname)
