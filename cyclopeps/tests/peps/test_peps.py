import unittest
from cyclopeps.tools.utils import *
import copy

class test_peps(unittest.TestCase):

    def test_normalization(self):
        from cyclopeps.tools.peps_tools import PEPS
        mpiprint(0,'\n'+'='*50+'\nPeps Normalization test\n'+'-'*50)
        Nx = 3
        Ny = 5
        d = 2
        D = 3
        chi = 10
        norm_tol = 1e-5
        peps = PEPS(Nx=Nx,Ny=Ny,d=d,D=D,chi=chi,norm_tol=norm_tol)
        norm = peps.calc_norm(chi=20) 
        print('Norm = {}'.format(norm))
        self.assertTrue(norm < 1./norm_tol)
        self.assertTrue(norm > norm_tol)
        mpiprint(0,'Passed\n'+'='*50)

if __name__ == "__main__":
    unittest.main()
