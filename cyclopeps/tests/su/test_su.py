import unittest
from cyclopeps.tools.utils import *
import copy

class test_cal_energy(unittest.TestCase):

    def test_energy_calc(self):
        from cyclopeps.tools.peps_tools import PEPS
        from cyclopeps.ops.itf import return_op
        from cyclopeps.algs.simple_update import run_tebd
        mpiprint(0,'\n'+'='*50+'\nSU Peps Normalization test\n'+'-'*50)
        # Create PEPS
        Nx = 5
        Ny = 5
        d = 2
        D = 2
        chi = 10
        Zn = None
        backend = 'numpy'
        # Get mpo
        ham = return_op(Nx,Ny,(1.,2.))
        # Run TEBD
        Ef,_ = run_tebd(Nx,
                        Ny,
                        d,
                        ham,
                        D=D,
                        chi=chi,
                        Zn=Zn,
                        backend=backend,
                        n_step=100)
        print('Final  E = {}'.format(Ef))

        mpiprint(0,'Passed\n'+'='*50)

    """
    def test_energy_calc_z2(self):
        from cyclopeps.tools.peps_tools import PEPS
        from cyclopeps.ops.itf import return_op
        from cyclopeps.algs.simple_update import run_tebd
        mpiprint(0,'\n'+'='*50+'\nSU Peps Normalization test (Z2 symmetry)\n'+'-'*50)
        # Create PEPS
        Nx = 5
        Ny = 5
        d = 2
        D = 2
        chi = 10
        Zn = 2
        backend = 'numpy'
        # Get mpo
        ham = return_op(Nx,Ny,(1.,2.))
        # Run TEBD
        Ef,_ = run_tebd(Nx,Ny,d,ham,D=D,chi=chi,n_step=100)
        print('Final  E = {}'.format(Ef))

        mpiprint(0,'Passed\n'+'='*50)
    """

if __name__ == "__main__":
    unittest.main()
