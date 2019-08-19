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

    def test_rotate(self):
        mpiprint(0,'\n'+'='*50+'\nPeps Rotation test\n'+'-'*50)
        from cyclopeps.tools.peps_tools import PEPS
        Nx = 3
        Ny = 3
        d = 2
        D = 3
        chi = 100
        peps = PEPS(Nx=Nx,Ny=Ny,d=d,D=D,chi=chi,normalize=False)
        norm0 = peps.calc_norm()
        peps.rotate()
        norm1 = peps.calc_norm()
        peps.rotate(clockwise=False)
        norm2 = peps.calc_norm()
        print('Norms = {},{},{}'.format(norm0,norm1,norm2))
        self.assertTrue(abs((norm0-norm1)/norm0) < 1e-5)
        self.assertTrue(abs((norm0-norm2)/norm0) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

    def test_flip(self):
        mpiprint(0,'\n'+'='*50+'\nPeps Flipping test\n'+'-'*50)
        from cyclopeps.tools.peps_tools import PEPS
        Nx = 5
        Ny = 5
        d = 2
        D = 3
        chi = 10
        peps = PEPS(Nx=Nx,Ny=Ny,d=d,D=D,chi=chi,normalize=False)
        norm0 = peps.calc_norm()
        peps.flip()
        norm1 = peps.calc_norm()
        peps.flip()
        norm2 = peps.calc_norm()
        print('Norms = {},{},{}'.format(norm0,norm1,norm2))
        self.assertTrue(abs((norm0-norm1)/norm0) < 1e-5)
        self.assertTrue(abs((norm0-norm2)/norm0) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

if __name__ == "__main__":
    unittest.main()
