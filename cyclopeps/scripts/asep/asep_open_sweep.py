from cyclopeps.tools.utils import *
from cyclopeps.tools.peps_tools import PEPS
from cyclopeps.ops.asep import return_op
from cyclopeps.algs.tebd import run_tebd
from sys import argv
from numpy import linspace

# Get input values
Nx = int(argv[1])
Ny = int(argv[2])
D  = int(argv[3])
#chi= int(argv[4])
d  = 2

# TEBD Parameters
step_sizes = [0.1,0.05, 0.01]
n_step =     [ 50,  50,   50]
chi        = [ 10,  20,   50]

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
