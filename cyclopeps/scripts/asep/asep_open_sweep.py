from cyclopeps.tools.utils import *
from cyclopeps.tools.peps_tools import PEPS
from cyclopeps.ops.asep import return_op,return_curr_op
from cyclopeps.ops.basic import return_dens_op
from cyclopeps.algs.tebd import run_tebd
from sys import argv
from numpy import linspace

# Get input values
Nx = int(argv[1])
Ny = int(argv[2])
D  = int(argv[3])
#chi= int(argv[4])
d  = 2
ns = 5

# TEBD Parameters
step_sizes = [0.5,0.1,0.05, 0.01]
n_step =     [  ns,ns,  ns,   ns]
chi        = [  5, 10,  20,   50]

# Sx parameters
sxVec = linspace(-0.5,0.5,20)
syVec = linspace(-0.5,0.5,20)
Ef = zeros((len(sxVec),len(syVec)))

# ---------------------------------------------------------
peps = peps = PEPS(Nx,Ny,d,D,chi[0])
for sxind in range(len(sxVec)):
    for syind in range(len(syVec)):
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

        # Create the Suzuki trotter decomposed operator
        ops = return_op(Nx,Ny,params)
        curr_ops = return_curr_op(Nx,Ny,params)
        dens_ops = return_dens_op(Nx,Ny)

        # Run TEBD
        try:
            Ef[sxind,syind],peps = run_tebd(Nx,
                                            Ny,
                                            d,
                                            ops,
                                            peps=peps,
                                            D=D,
                                            chi=chi,
                                            n_step=n_step,
                                            step_size=step_sizes)
        except:
            try:
                peps = PEPS(Nx,Ny,d,D,chi[0])
                Ef[sxind,syind],peps = run_tebd(Nx,
                                                Ny,
                                                d,
                                                ops,
                                                peps=peps,
                                                D=D,
                                                chi=chi,
                                                n_step=n_step,
                                                step_size=step_sizes)
            except:
                peps = None

        # Evaluate X and Y Current and local density
        if peps is not None:
            # Calculate Current
            currents = peps.calc_op(curr_ops,return_sum=False)
            print('Vertical Currents')
            for i in range(Nx):
                print_str = ''
                for j in range(Ny-1):
                    print_str += '{} '.format(currents[0][i][j])
                print(print_str)
            print('Horizontal Currents')
            for i in range(Ny):
                print_str = ''
                for j in range(Nx-1):
                    print_str += '{} '.format(currents[1][i][j])
                print(print_str)
            # Calculate Density
            density = peps.calc_op(dens_ops,return_sum=False)
            print('Vertical Density')
            for i in range(Nx):
                print_str = ''
                for j in range(Ny-1):
                    print_str += '{} '.format(density[0][i][j])
                print(print_str)
            print('Horizontal Density')
            for i in range(Ny):
                print_str = ''
                for j in range(Nx-1):
                    print_str += '{} '.format(density[1][i][j])
                print(print_str)

        # Print sx
        print_str = '\t'
        for i1 in range(len(sxVec)):
            print_str += '\t{}'.format(sxVec[i1])
        print(print_str)
        # print sy results
        for i1 in range(len(syVec)):
            print_str = '{}\t'.format(syVec[i1])
            for i2 in range(len(sxVec)):
                print_str += '{}\t'.format(Ef[i2,i1])
            print(print_str)
