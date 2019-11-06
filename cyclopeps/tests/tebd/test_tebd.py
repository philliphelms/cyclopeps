import unittest
from cyclopeps.tools.utils import *
import copy

class test_cal_energy(unittest.TestCase):

    """
    def test_energy_calc_itf(self):
        from cyclopeps.tools.peps_tools import PEPS
        from cyclopeps.ops.itf import return_op
        from cyclopeps.algs.tebd import run_tebd
        mpiprint(0,'\n'+'='*50+'\nPeps Full Update ITF test\n'+'-'*50)
        # Create PEPS
        Nx = 5
        Ny = 5
        d = 2
        D = 2
        chi = 20
        # Get mpo
        ham = return_op(Nx,Ny,(1.,2.))
        # Run TEBD
        Ef,_ = run_tebd(Nx,Ny,d,ham,D=D,chi=chi,n_step=20)
        print('Final  E = {}'.format(Ef))
        mpiprint(0,'Passed\n'+'='*50)
    """

    def test_energy_calc_asep(self):
        from cyclopeps.tools.peps_tools import PEPS
        from cyclopeps.ops.asep import return_op
        from cyclopeps.algs.tebd import run_tebd
        mpiprint(0,'\n'+'='*50+'\nPeps Full Update ASEP test\n'+'-'*50)
        # Create PEPS
        Nx = 2
        Ny = 2
        d = 2
        D = 2
        chi = 20
        # Get mpo
        params = (0.9,0.1,0.9,0.1,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5)
        #params = (0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.,0.)
        ham = return_op(Nx,Ny,params)
        # Run TEBD
        Ef,_ = run_tebd(Nx,Ny,d,ham,D=D,chi=chi,n_step=20,su_step_size=[1,0.1,0.01])
        print('Final  E = {}'.format(Ef))
        mpiprint(0,'Passed\n'+'='*50)

    """
    def test_energy_calc_east(self):
        from cyclopeps.tools.peps_tools import PEPS
        from cyclopeps.ops.east import return_op
        from cyclopeps.algs.tebd import run_tebd
        mpiprint(0,'\n'+'='*50+'\nPeps Full Update East test\n'+'-'*50)
        # Create PEPS
        Nx = 4
        Ny = 4
        d = 2
        D = 2
        chi = 20
        # Get mpo
        params = (0.2,-0.5)
        ham = return_op(Nx,Ny,params)
        # Run TEBD
        Ef,_ = run_tebd(Nx,Ny,d,ham,D=D,chi=chi,n_step=20)
        print('Final  E = {}'.format(Ef))
        mpiprint(0,'Passed\n'+'='*50)

    def test_energy_calc_east2(self):
        from cyclopeps.tools.peps_tools import PEPS
        from cyclopeps.ops.east import return_op
        from cyclopeps.algs.tebd import run_tebd
        mpiprint(0,'\n'+'='*50+'\nPeps Full Update East test\n'+'-'*50)
        # Create PEPS
        Nx = 2
        Ny = 2
        d = 2
        D = 2
        chi = 20
        # Get mpo
        params = (0.2,0.5)
        ham = return_op(Nx,Ny,params)
        # Run TEBD
        Ef,_ = run_tebd(Nx,Ny,d,ham,D=D,chi=chi,n_step=20)
        print('Final  E = {}'.format(Ef))
        mpiprint(0,'Passed\n'+'='*50)
    """

if __name__ == "__main__":
    unittest.main()
