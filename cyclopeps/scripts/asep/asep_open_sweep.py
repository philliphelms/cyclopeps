from cyclopeps.tools.utils import *
from cyclopeps.tools.peps_tools import PEPS
from cyclopeps.tools.ops_tools import ops_conj_trans
from cyclopeps.ops.asep import return_op,return_curr_op
from cyclopeps.ops.basic import return_dens_op
from cyclopeps.algs.tebd import run_tebd
from sys import argv
from numpy import linspace

# Get input values
sxind = int(argv[3])
syind = int(argv[4])

# Calculation Parameters
Nx = int(argv[1])
Ny = int(argv[2])
d  = 2

## Calculation parameters
#dt = [ 0.1,  0.05,  0.1,  0.05, 0.01,  0.1, 0.01,0.001,  0.1, 0.05, 0.01,0.001]
#D  = [   2,     2,    4,    4,    4,    6,    6,    6,    8,    8,    8,    8]
#chi= [   5,    10,   10,   20,   30,   30,   40,   50,   30,   40,   50,   60]
#ns = [   5,     5,   20,   20,   20,   20,   20,   20,   20,   20,   20,   20]
#conv=[1e-2,  1e-3, 1e-2, 1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 1e-3, 1e-4, 1e-4, 1e
dt = [ 0.1,  0.05,  0.1,  0.1, 0.05, 0.05, 0.01,  0.1,  0.1, 0.01, 0.01, 0.001,  0.1,  0.1, 0.05, 0.05, 0.01, 0.01, 0.001]
D  = [   2,     2,    4,    4,    4,    4,    4,    6,    6,    6,    6,     6,    8,    8,    8,    8,    8,    8,     8]
chi= [   5,    10,   10,   20,   20,   30,   30,   30,   40,   40,   50,    50,   30,   40,   40,   50,   50,   60,    60]
ns = [   5,     5,   15,   15,   15,   15,   20,   15,   15,   15,   15,    20,   15,   15,   15,   15,   15,   15,    20]
conv=[1e-2,  1e-3, 1e-2, 1e-3, 1e-3, 1e-3, 1e-4, 1e-3, 1e-3, 1e-4, 1e-4,  1e-5, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5,  1e-6]

# Sx parameters
sxVec = linspace(-0.5,1.,16)
syVec = linspace(-0.5,1.,16)

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
peps = PEPS(Nx,Ny,d,D[0],chi[0])
pepsl = PEPS(Nx,Ny,d,D[0],chi[0])
for ind in range(len(D)):
    print('Right Vector' + '-'*50)
    # Calculate right state
    try:
        Ef,peps = run_tebd(Nx,
                           Ny,
                           d,
                           ops,
                           peps=peps,
                           D=D[ind],
                           chi=chi[ind],
                           n_step=ns[ind],
                           step_size=dt[ind])
    except Exception as e: 
        print('Error in using previous peps, starting with random')
        print(e)
        try:
            peps = PEPS(Nx,Ny,d,D[ind],chi[ind])
            Ef,peps = run_tebd(Nx,
                               Ny,
                               d,
                               ops,
                               peps=peps,
                               D=D[ind],
                               chi=chi[ind],
                               n_step=ns[ind],
                               step_size=dt[ind])
        except Exception as e: 
            print('Error in random start, skipping calculation')
            print(e)
            peps = PEPS(Nx,Ny,d,D[ind],chi[ind])
    # Calculate left state
    print('Left  Vector' + '-'*50)
    try:
        Efl,pepsl = run_tebd(Nx,
                           Ny,
                           d,
                           ops,
                           peps=pepsl,
                           D=D[ind],
                           chi=chi[ind],
                           n_step=ns[ind],
                           step_size=dt[ind])
    except Exception as e: 
        print('Error in using previous peps, starting with random')
        print(e)
        try:
            pepsl = PEPS(Nx,Ny,d,D[ind],chi[ind])
            Efl,pepsl = run_tebd(Nx,
                               Ny,
                               d,
                               ops,
                               peps=pepsl,
                               D=D[ind],
                               chi=chi[ind],
                               n_step=ns[ind],
                               step_size=dt[ind])
        except Exception as e: 
            print('Error in random start, skipping calculation')
            print(e)
            pepsl = PEPS(Nx,Ny,d,D[ind],chi[ind])


    # Evaluate X and Y Current and local density
    # Calculate Current
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

    # Increase PEPS D if wanted
    if (len(D)-1 > ind):
        if (D[ind+1] > D[ind]):
            peps.increase_mbd(D[ind+1],chi=chi[ind+1])
        peps.chi = chi[ind+1]
        if (D[ind+1] > D[ind]):
            pepsl.increase_mbd(D[ind+1],chi=chi[ind+1])
        pepsl.chi = chi[ind+1]
        peps.normalize()
