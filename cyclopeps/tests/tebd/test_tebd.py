import unittest
from cyclopeps.tools.utils import *
import copy

class test_cal_energy(unittest.TestCase):

    def test_energy_calc(self):
        from cyclopeps.tools.peps_tools import PEPS
        from cyclopeps.ops.itf import return_op
        from cyclopeps.algs.tebd import run_tebd
        mpiprint(0,'\n'+'='*50+'\nPeps Normalization test\n'+'-'*50)
        # Create PEPS
        Nx = 3
        Ny = 3
        d = 2
        D = 3
        chi = 10
        norm_tol = 1e-5
        # Get mpo
        ham = return_op(Nx,Ny,(1.,2.))
        # Run TEBD
        Ef = run_tebd(Nx,Ny,d,ham,D=D,chi=chi)
        print('Final  E = {}'.format(Ef))

        mpiprint(0,'Passed\n'+'='*50)

if __name__ == "__main__":
    unittest.main()
