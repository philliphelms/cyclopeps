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

## Calculation parameters
dt     = [0.1,0.01]+[0.01]*20
D      = [1]*2+[2]*4+[4]*4+[6]*4+[8]*4+[10]*4
chi    = [1]*2+[20,40,60,80]*5
conv   = [1e-6]*22
n_step = [1000,1000]+[500]*20
d  = 2

# Sx parameters
sxVec = linspace(-0.5,1.,16)
syVec = linspace(-0.5,1.,16)

# Filenames for saved PEPS
savedir = "./saved_peps/asep/"
fnamel = "Nx{}_Ny{}_sx{}_sy{}_run_left".format(Nx,Ny,sxind,syind)
fnamer = "Nx{}_Ny{}_sx{}_sy{}_run_right".format(Nx,Ny,sxind,syind)

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
opsl= ops_conj_trans(ops)
curr_ops = return_curr_op(Nx,Ny,params)
dens_ops_top = return_dens_op(Nx,Ny,top=True)
dens_ops_bot = return_dens_op(Nx,Ny,top=False)

# Run TEBD
peps = PEPS(Nx,Ny,d,D[0],chi[0],fname=fnamer,fdir=savedir)
pepsl = PEPS(Nx,Ny,d,D[0],chi[0],fname=fnamel,fdir=savedir)

# Loop over all optimizaton parameters
for ind in range(len(D)):

    # --------------------------------------------------------------------
    # Calculate right eigenstate
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

    # --------------------------------------------------------------------
    # Calculate left eigenstate
    Efl,pepsl = run_tebd(Nx,
                       Ny,
                       d,
                       ops,
                       peps=pepsl,
                       D=D[ind],
                       chi=chi[ind],
                       n_step=n_step[ind],
                       step_size=dt[ind],
                       conv_tol=conv[ind],
                       print_prepend = '(left) ')

    # --------------------------------------------------------------------
    # Evaluate Operators
    # Current
    currents = peps.calc_op(curr_ops,return_sum=False,ket=pepsl)
    print('Vertical Currents = {}'.format(currents[0].sum()))
    for i in range(Nx):
        print_str = ''
        for j in range(Ny-1):
            print_str += '{} '.format(currents[0][i][j])
        print(print_str)
    print('Horizontal Currents = {}'.format(currents[1].sum()))
    for i in range(Ny):
        print_str = ''
        for j in range(Nx-1):
            print_str += '{} '.format(currents[1][i][j])
        print(print_str)
    # Calculate Density
    density_top = peps.calc_op(dens_ops_top,return_sum=False,ket=pepsl)
    density_bot = peps.calc_op(dens_ops_bot,return_sum=False,ket=pepsl)
    print('Vertical Density')
    for i in range(Nx):
        print_str = ''
        for j in range(Ny-1):
            print_str += '{} '.format(density_top[0][i][j])
        print(print_str)
    for i in range(Nx):
        print_str = ''
        for j in range(Ny-1):
            print_str += '{} '.format(density_bot[0][i][j])
        print(print_str)
    print('Horizontal Density')
    for i in range(Ny):
        print_str = ''
        for j in range(Nx-1):
            print_str += '{} '.format(density_top[1][i][j])
        print(print_str)
    for i in range(Ny):
        print_str = ''
        for j in range(Nx-1):
            print_str += '{} '.format(density_bot[1][i][j])
        print(print_str)
