import unittest
from cyclopeps.tools.utils import *
import copy

class test_peps(unittest.TestCase):

    def test_flip(self):
        mpiprint(0,'\n'+'='*50+'\nPeps Flipping test\n'+'-'*50)
        from cyclopeps.tools.peps_tools import PEPS
        Nx = 5
        Ny = 5
        d = 2
        D = 3
        chi = 10
        peps = PEPS(Nx=Nx,Ny=Ny,d=d,D=D,chi=chi,normalize=False,backend='ctf')
        norm0 = peps.calc_norm()
        peps.flip()
        norm1 = peps.calc_norm()
        peps.flip()
        norm2 = peps.calc_norm()
        mpiprint(0,'Norms = {},{},{}'.format(norm0,norm1,norm2))
        self.assertTrue(abs((norm0-norm1)/norm0) < 1e-5)
        self.assertTrue(abs((norm0-norm2)/norm0) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

    def test_normalization(self):
        from cyclopeps.tools.peps_tools import PEPS
        mpiprint(0,'\n'+'='*50+'\nPeps (5x5) Normalization test without Symmetry\n'+'-'*50)
        Nx  = 5
        Ny  = 5
        d   = 2
        D   = 6
        chi = 10
        Zn  = None
        backend  = 'ctf'
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

    def test_normalization_Z3(self):
        from cyclopeps.tools.peps_tools import PEPS
        mpiprint(0,'\n'+'='*50+'\nPeps (5x5) Normalization test with Z3 Symmetry\n'+'-'*50)
        Nx  = 5
        Ny  = 5
        d   = 2
        D   = 6
        chi = 50
        Zn  = 3 # Zn symmetry (here, Z3)
        dZn = 2
        backend  = 'ctf'
        # Generate random PEPS
        peps = PEPS(Nx=Nx,
                    Ny=Ny,
                    d=d,
                    D=D,
                    chi=chi,
                    Zn=Zn,
                    dZn=dZn,
                    backend=backend,
                    normalize=False)
        # Compute the norm (2 ways for comparison)
        peps_sparse = peps.make_sparse()
        norm0 = peps.calc_norm(chi=chi) 
        norm1 = peps_sparse.calc_norm(chi=chi)
        mpiprint(0,'Symmetric Sparse Norm = {}'.format(norm1))
        mpiprint(0,'Symmetric Dense Norm = {}'.format(norm0))
        self.assertTrue(abs((norm0-norm1)/norm1) < 1e-3)
        mpiprint(0,'Passed\n'+'='*50)

    def test_normalization_large_Z2(self):
        from cyclopeps.tools.peps_tools import PEPS
        mpiprint(0,'\n'+'='*50+'\nPeps (10x10) Normalization test with Z2 Symmetry\n'+'-'*50)
        Nx  = 10
        Ny  = 10
        d   = 2
        D   = 6
        chi = 10
        Zn  = 2 # Zn symmetry (here, Z2)
        backend  = 'ctf'
        # Generate random PEPS
        peps = PEPS(Nx=Nx,
                    Ny=Ny,
                    d=d,
                    D=D,
                    chi=chi,
                    Zn=Zn,
                    backend=backend,
                    normalize=False)
        # Compute the norm (2 ways for comparison)
        norm0 = peps.calc_norm(chi=chi) 
        mpiprint(0,'Symmetric Dense Norm = {}'.format(norm0))
        peps_sparse = peps.make_sparse()
        norm1 = peps_sparse.calc_norm(chi=chi)
        mpiprint(0,'Symmetric Sparse Norm = {}'.format(norm1))
        # Normalize the PEPS
        norm2 = peps.normalize()
        mpiprint(0,'Symmetric Dense Norm (After normalized) = {}'.format(norm2))
        peps_sparse = peps.make_sparse()
        norm3 = peps_sparse.calc_norm(chi=chi)
        mpiprint(0,'Symmetric Sparse Norm (After normalized) = {}'.format(norm3))
        # Do some assertions to check if passed
        self.assertTrue(abs((norm0-norm1)/norm1) < 1e-3)
        self.assertTrue(abs(1.0-norm2) < 1e-3)
        self.assertTrue(abs(1.0-norm3) < 1e-3)
        mpiprint(0,'Passed\n'+'='*50)

    def test_normalization_Z2(self):
        from cyclopeps.tools.peps_tools import PEPS
        mpiprint(0,'\n'+'='*50+'\nPeps (5x5) Normalization test with Z2 Symmetry\n'+'-'*50)
        Nx  = 5
        Ny  = 5
        d   = 2
        D   = 6
        chi = 10
        Zn  = 2 # Zn symmetry (here, Z2)
        backend  = 'ctf'
        # Generate random PEPS
        peps = PEPS(Nx=Nx,
                    Ny=Ny,
                    d=d,
                    D=D,
                    chi=chi,
                    Zn=Zn,
                    backend=backend,
                    normalize=False)
        # Compute the norm (2 ways for comparison)
        norm0 = peps.calc_norm(chi=chi) 
        peps_sparse = peps.make_sparse()
        norm1 = peps_sparse.calc_norm(chi=chi)
        mpiprint(0,'Symmetric Dense Norm = {}'.format(norm0))
        mpiprint(0,'Symmetric Sparse Norm = {}'.format(norm1))
        # Normalize the PEPS
        norm2 = peps.normalize()
        peps_sparse = peps.make_sparse()
        norm3 = peps_sparse.calc_norm(chi=chi)
        mpiprint(0,'Symmetric Dense Norm (After normalized) = {}'.format(norm2))
        mpiprint(0,'Symmetric Sparse Norm (After normalized) = {}'.format(norm3))
        # Do some assertions to check if passed
        self.assertTrue(abs((norm0-norm1)/norm1) < 1e-3)
        self.assertTrue(abs(1.0-norm2) < 1e-3)
        self.assertTrue(abs(1.0-norm3) < 1e-3)
        mpiprint(0,'Passed\n'+'='*50)


    def test_rotate(self):
        mpiprint(0,'\n'+'='*50+'\nPeps Rotation test\n'+'-'*50)
        from cyclopeps.tools.peps_tools import PEPS
        Nx = 3
        Ny = 3
        d = 2
        D = 3
        chi = 100
        peps = PEPS(Nx=Nx,Ny=Ny,d=d,D=D,chi=chi,normalize=False,backend='ctf')
        norm0 = peps.calc_norm()
        peps.rotate()
        norm1 = peps.calc_norm()
        peps.rotate(clockwise=False)
        norm2 = peps.calc_norm()
        mpiprint(0,'Norms = {},{},{}'.format(norm0,norm1,norm2))
        self.assertTrue(abs((norm0-norm1)/norm0) < 1e-5)
        self.assertTrue(abs((norm0-norm2)/norm0) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)


    def test_normalization_Z2(self):
        from cyclopeps.tools.peps_tools import PEPS
        mpiprint(0,'\n'+'='*50+'\nPeps (5x5) Normalization test with Z2 Symmetry\n'+'-'*50)
        Nx  = 5
        Ny  = 5
        d   = 2
        D   = 6
        chi = 10
        Zn  = 2 # Zn symmetry (here, Z2)
        backend  = 'ctf'
        # Generate random PEPS
        peps = PEPS(Nx=Nx,
                    Ny=Ny,
                    d=d,
                    D=D,
                    chi=chi,
                    Zn=Zn,
                    backend=backend,
                    normalize=False)
        # Compute the norm (2 ways for comparison)
        norm0 = peps.calc_norm(chi=chi) 
        peps_sparse = peps.make_sparse()
        norm1 = peps_sparse.calc_norm(chi=chi)
        mpiprint(0,'Symmetric Dense Norm = {}'.format(norm0))
        mpiprint(0,'Symmetric Sparse Norm = {}'.format(norm1))
        # Normalize the PEPS
        norm2 = peps.normalize()
        peps_sparse = peps.make_sparse()
        norm3 = peps_sparse.calc_norm(chi=chi)
        mpiprint(0,'Symmetric Dense Norm (After normalized) = {}'.format(norm2))
        mpiprint(0,'Symmetric Sparse Norm (After normalized) = {}'.format(norm3))
        # Do some assertions to check if passed
        self.assertTrue(abs((norm0-norm1)/norm1) < 1e-3)
        self.assertTrue(abs(1.0-norm2) < 1e-3)
        self.assertTrue(abs(1.0-norm3) < 1e-3)
        mpiprint(0,'Passed\n'+'='*50)

    def test_rotate_Z2(self):
        mpiprint(0,'\n'+'='*50+'\nPeps Z2 Rotation test\n'+'-'*50)
        from cyclopeps.tools.peps_tools import PEPS
        Nx = 5
        Ny = 5
        d = 2
        D = 6
        Zn = 2
        chi = 10
        backend  = 'ctf'
        # Generate random PEPS
        peps = PEPS(Nx=Nx,
                    Ny=Ny,
                    d=d,
                    D=D,
                    chi=chi,
                    Zn=Zn,
                    backend=backend,
                    normalize=False)
        # Compute the norm (2 ways for comparison)
        norm0 = peps.calc_norm(chi=chi) 
        peps_sparse = peps.make_sparse()
        norm1 = peps_sparse.calc_norm(chi=chi)
        mpiprint(0,'Symmetric Dense Norm = {}'.format(norm0))
        mpiprint(0,'Symmetric Sparse Norm = {}'.format(norm1))
        # Rotate the PEPS
        peps.rotate(clockwise=False)
        norm2 = peps.calc_norm(chi=chi) 
        peps_sparse = peps.make_sparse()
        norm3 = peps_sparse.calc_norm(chi=chi)
        mpiprint(0,'Rotated Symmetric Dense Norm = {}'.format(norm2))
        mpiprint(0,'Rotated Symmetric Sparse Norm = {}'.format(norm3))
        self.assertTrue(abs((norm0-norm1)/norm1) < 1e-3)
        self.assertTrue(abs((norm0-norm2)/norm2) < 1e-3)
        self.assertTrue(abs((norm0-norm3)/norm3) < 1e-3)
        mpiprint(0,'Passed\n'+'='*50)

    def test_flip_Z2(self):
        mpiprint(0,'\n'+'='*50+'\nPeps Z2 Flipping test\n'+'-'*50)
        from cyclopeps.tools.peps_tools import PEPS
        Nx = 5
        Ny = 5
        d = 2
        D = 6
        Zn = 2
        chi = 10
        backend  = 'ctf'
        # Generate random PEPS
        peps = PEPS(Nx=Nx,
                    Ny=Ny,
                    d=d,
                    D=D,
                    chi=chi,
                    Zn=Zn,
                    backend=backend,
                    normalize=False)
        # Compute the norm (2 ways for comparison)
        norm0 = peps.calc_norm(chi=chi) 
        peps_sparse = peps.make_sparse()
        norm1 = peps_sparse.calc_norm(chi=chi)
        mpiprint(0,'Symmetric Dense Norm = {}'.format(norm0))
        mpiprint(0,'Symmetric Sparse Norm = {}'.format(norm1))
        # Rotate the PEPS
        peps.flip()
        norm2 = peps.calc_norm(chi=chi) 
        peps_sparse = peps.make_sparse()
        norm3 = peps_sparse.calc_norm(chi=chi)
        mpiprint(0,'Flipped Symmetric Dense Norm = {}'.format(norm2))
        mpiprint(0,'Flipped Symmetric Sparse Norm = {}'.format(norm3))
        self.assertTrue(abs((norm0-norm1)/norm1) < 1e-3)
        self.assertTrue(abs((norm0-norm2)/norm2) < 1e-3)
        self.assertTrue(abs((norm0-norm3)/norm3) < 1e-3)
        mpiprint(0,'Passed\n'+'='*50)

if __name__ == "__main__":
    unittest.main()
