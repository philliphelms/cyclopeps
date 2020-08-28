from cyclopeps.tools.utils import *
from cyclopeps.tools.peps_tools import PEPS
from cyclopeps.tools.ops_tools import ops_conj_trans
from cyclopeps.ops.asep import return_op,return_curr_op
from cyclopeps.ops.basic import return_dens_op
from cyclopeps.algs.tebd import run_tebd
from sys import argv
from numpy import linspace
import numpy as np

# Get Number of sites
Nvec = [int(argv[1])]

# PEPS Directory
pepsdir = '/central/groups/changroup/members/phelms/asep/final/peps/'

# Sx parameters
sxVec = linspace(-0.5,0.5,21)
syVec = linspace(-0.5,0.5,21)

# Convergence Parameters
prepend_vec = ['final','conv']
D   = [1]+[2]*6+[3]*6+[4]*6+[5]*6+[6]*6+[7]*6+[8]*6+[9]*6+[10]*6
chi = [1]+[20,40,60,80,100,150]*9
d = 2

for N in Nvec:
    for prepend in prepend_vec:
        for dind in range(len(D)):
            for sxind in range(21)[::-1]:

                loaded = False
                # Get peps loaded
                try:
                    fnamel = prepend+'Nx{}_Ny{}_sx{}_sy{}_D{}_chi{}_run_left'.format(N,N,sxind,0,D[dind],chi[dind])
                    pepsl = PEPS(N,N,d,D[dind],chi[dind],
                                 norm_tol=0.5,
                                 norm_bs_upper=3.,
                                 norm_bs_lower=0.,
                                 normalize=False)
                    pepsl.load_tensors(pepsdir+fnamel)
                    loaded=True
                except Exception as e:
                    #print('Failed to get PEPS left loaded: {}'.format(e))
                    pepsl = None
                try:
                    fnamer = prepend+'Nx{}_Ny{}_sx{}_sy{}_D{}_chi{}_run_right'.format(N,N,sxind,0,D[dind],chi[dind])
                    peps = PEPS(N,N,d,D[dind],chi[dind],
                                norm_tol=0.5,
                                norm_bs_upper=3.,
                                norm_bs_lower=0.,
                                normalize=False)
                    peps.load_tensors(pepsdir+fnamer)
                    loaded=True
                except Exception as e:
                    #print('Failed to get PEPS right loaded: {}'.format(e))
                    peps = None

                if loaded:
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
                    sy = syVec[0]
                    params = (jr,jl,ju,jd,cr,cl,cu,cd,dr,dl,du,dd,sx,sy)

                    # Create the Suzuki trotter decomposed operator
                    ops = return_op(N,N,params)
                    opsl= ops_conj_trans(ops)
                    curr_ops = return_curr_op(N,N,params)
                    dens_ops_top = return_dens_op(N,N,top=True)
                    dens_ops_bot = return_dens_op(N,N,top=False)

                    try:
                        E = peps.calc_op(ops,return_sum=True,chi=chi[dind])
                        El = pepsl.calc_op(opsl,return_sum=True,chi=chi[dind])
                        Elr = peps.calc_op(ops,return_sum=True,ket=pepsl,chi=chi[dind])
                        currents = peps.calc_op(curr_ops,return_sum=False,ket=pepsl,chi=chi[dind])
                        density_top = peps.calc_op(dens_ops_top,return_sum=False,ket=pepsl,chi=chi[dind])
                        print('{},{},{},{},{},{},{},{},{},{}'.format(N,D[dind],chi[dind],sxind,E,El,Elr,currents[0].sum(),currents[1].sum(),np.average(density_top[0])))


                        #print('Vertical Currents = {}'.format(currents[0].sum()))
                        #for i in range(N):
                        #    print_str = ''
                        #    for j in range(N-1):
                        #        print_str += '{} '.format(currents[0][i][j])
                        #    print(print_str)
                        #print('Horizontal Currents = {}'.format(currents[1].sum()))
                        #for i in range(N):
                        #    print_str = ''
                        #    for j in range(N-1):
                        #        print_str += '{} '.format(currents[1][i][j])
                        #    print(print_str)
                        # Calculate Density
                        #density_bot = peps.calc_op(dens_ops_bot,return_sum=False,ket=pepsl,chi=chi[dind])
                        #print('Average Density = {}'.format(np.average(density_top[0])))
                        #for i in range(N):
                        #    print_str = ''
                        #    for j in range(N-1):
                        #        print_str += '{} '.format(density_top[0][i][j])
                        #    print(print_str)
                        #for i in range(N):
                        #    print_str = ''
                        #    for j in range(N-1):
                        #        print_str += '{} '.format(density_bot[0][i][j])
                        #    print(print_str)
                        #print('Horizontal Density')
                        #for i in range(N):
                        #    print_str = ''
                        #    for j in range(N-1):
                        #        print_str += '{} '.format(density_top[1][i][j])
                        #    print(print_str)
                        #for i in range(N):
                        #    print_str = ''
                        #    for j in range(N-1):
                        #        print_str += '{} '.format(density_bot[1][i][j])
                        #    print(print_str)
                    except Exception as e:
                        print('Failed to compute operators: {}'.format(e))
