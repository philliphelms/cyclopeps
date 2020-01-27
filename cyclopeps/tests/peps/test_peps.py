import unittest
from cyclopeps.tools.utils import *
import copy

class test_peps(unittest.TestCase):

    def test_normalization_Z2(self):
        from cyclopeps.tools.peps_tools import PEPS
        mpiprint(0,'\n'+'='*50+'\nPeps Normalization test with Z2 Symmetry\n'+'-'*50)
        Nx  = 3
        Ny  = 3
        d   = 2
        D   = 6
        chi = 10
        Zn  = 2 # Zn symmetry (here, Z2)
        backend  = 'numpy'
        peps = PEPS(Nx=Nx,
                    Ny=Ny,
                    d=d,
                    D=D,
                    chi=chi,
                    Zn=Zn,
                    backend=backend,
                    normalize=True)
        norm = peps.calc_norm(chi=chi) 
        mpiprint(0,'Norm = {}'.format(norm))
        self.assertTrue(abs(1.0-norm) < 1e-3)
        mpiprint(0,'Passed\n'+'='*50)

    def test_normalization(self):
        from cyclopeps.tools.peps_tools import PEPS
        mpiprint(0,'\n'+'='*50+'\nPeps Normalization test without Symmetry\n'+'-'*50)
        Nx  = 3
        Ny  = 3
        d   = 2
        D   = 6
        chi = 10
        Zn  = None
        backend  = 'numpy'
        peps = PEPS(Nx=Nx,
                    Ny=Ny,
                    d=d,
                    D=D,
                    chi=chi,
                    Zn=Zn,
                    backend=backend,
                    normalize=True)
        norm = peps.calc_norm(chi=chi) 
        mpiprint(0,'Norm = {}'.format(norm))
        self.assertTrue(abs(1.0-norm) < 1e-3)
        mpiprint(0,'Passed\n'+'='*50)

    '''
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
        mpiprint(0,'Norms = {},{},{}'.format(norm0,norm1,norm2))
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
        mpiprint(0,'Norms = {},{},{}'.format(norm0,norm1,norm2))
        self.assertTrue(abs((norm0-norm1)/norm0) < 1e-5)
        self.assertTrue(abs((norm0-norm2)/norm0) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)
    '''

if __name__ == "__main__":
    unittest.main()
