import unittest
from cyclopeps.tools.utils import *
import copy

class test_cal_energy(unittest.TestCase):

    def test_energy_calc(self):
        from cyclopeps.tools.peps_tools import PEPS
        from cyclopeps.ops.itf import return_op
        mpiprint(0,'\n'+'='*50+'\nPeps Normalization test\n'+'-'*50)
        # Create PEPS
        Nx = 3
        Ny = 5
        d = 2
        D = 3
        chi = 10
        norm_tol = 1e-5
        peps = PEPS(Nx=Nx,Ny=Ny,d=d,D=D,chi=chi,norm_tol=norm_tol)
        # Get ops
        ham = return_op(Nx,Ny,(1.,2.))
        norm = peps.calc_norm(chi=20) 
        E = peps.calc_op(ham)
        print('E = {}'.format(E/norm))
        mpiprint(0,'Passed\n'+'='*50)

    def test_energy_contraction_ones(self):
        mpiprint(0,'\n'+'='*50+'\nPeps Energy (Ham=Identity, peps=ones) calculation\n'+'-'*50)
        # Create a PEPS
        from cyclopeps.tools.peps_tools import PEPS
        Nx = 2
        Ny = 2
        d = 2
        D = 3
        peps = PEPS(Nx=Nx,Ny=Ny,d=d,D=D,chi=1000)
        # Set all tensor values to 1
        for xind in range(Nx):
            for yind in range(Ny):
                peps[xind][yind] = ones(peps[xind][yind].shape,peps[xind][yind].dtype)
        # Get the Hamiltonian
        from cyclopeps.ops.identity import return_op
        ham = return_op(Nx,Ny)
        # Calculate initial norm
        norm0 = peps.calc_norm()
        print('Norm = {}'.format(norm0))
        # Perform the Exact energy calculation:
        bra = einsum('LDWCM,lMXcu,CdYRm,cmZru->WXYZ',peps[0][0],peps[0][1],peps[1][0],peps[1][1])
        norm1 = einsum('WXYZ,WXYZ->',bra,conj(bra))
        E1  = einsum('WXYZ,wxYZ,WXwx->',bra,conj(bra),ham[0][0][0])
        E1 += einsum('WXYZ,wXyZ,WYwy->',bra,conj(bra),ham[1][0][0])
        E1 += einsum('WXYZ,WXyz,YZyz->',bra,conj(bra),ham[0][1][0])
        E1 += einsum('WXYZ,WxYz,XZxz->',bra,conj(bra),ham[1][1][0])
        # Contract Energy again
        E2 = peps.calc_op(ham,normalize=False)
        print('Energy = {}'.format(E2))
        self.assertTrue(abs((norm0-norm1)/norm0) < 1e-10)
        self.assertTrue(abs((norm0-E1)/norm0) < 1e-10)
        self.assertTrue(abs((norm0-E2)/norm0) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

    def test_energy_itf_contraction_ones(self):
        mpiprint(0,'\n'+'='*50+'\nPeps Energy (Ham=ITF, peps=ones) calculation\n'+'-'*50)
        # Create a PEPS
        from cyclopeps.tools.peps_tools import PEPS
        Nx = 2
        Ny = 2
        d = 2
        D = 3
        peps = PEPS(Nx=Nx,Ny=Ny,d=d,D=D,chi=1000)
        # Set all tensor values to 1
        for xind in range(Nx):
            for yind in range(Ny):
                peps[xind][yind] = ones(peps[xind][yind].shape,peps[xind][yind].dtype)
        # Get the Hamiltonian
        from cyclopeps.ops.itf import return_op
        ham = return_op(Nx,Ny,(1.,2.))
        # Calculate initial norm
        norm0 = peps.calc_norm()
        print('Norm = {}'.format(norm0))
        # Perform the Exact energy calculation:
        bra = einsum('LDWCM,lMXcu,CdYRm,cmZru->WXYZ',peps[0][0],peps[0][1],peps[1][0],peps[1][1])
        norm1 = einsum('WXYZ,WXYZ->',bra,conj(bra))
        E1  = einsum('WXYZ,wxYZ,WXwx->',bra,conj(bra),ham[0][0][0])
        E1 += einsum('WXYZ,wXyZ,WYwy->',bra,conj(bra),ham[1][0][0])
        E1 += einsum('WXYZ,WXyz,YZyz->',bra,conj(bra),ham[0][1][0])
        E1 += einsum('WXYZ,WxYz,XZxz->',bra,conj(bra),ham[1][1][0])
        # Contract Energy again
        E2 = peps.calc_op(ham,normalize=False)
        print('Energy (exact)   = {}'.format(E1))
        print('Energy (routine) = {}'.format(E2))
        self.assertTrue(abs((E2-E1)/E1) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

    def test_energy_contraction(self):
        mpiprint(0,'\n'+'='*50+'\nPeps Energy (Ham=Identity) calculation\n'+'-'*50)
        # Create a PEPS
        from cyclopeps.tools.peps_tools import PEPS
        Nx = 2
        Ny = 2
        d = 2
        D = 3
        peps = PEPS(Nx=Nx,Ny=Ny,d=d,D=D,chi=1000)
        # Get the Hamiltonian
        from cyclopeps.ops.identity import return_op
        ham = return_op(Nx,Ny)
        # Calculate initial norm
        norm0 = peps.calc_norm()
        # Perform the Exact energy calculation:
        bra = einsum('LDWCM,lMXcu,CdYRm,cmZru->WXYZ',peps[0][0],peps[0][1],peps[1][0],peps[1][1])
        norm1 = einsum('WXYZ,WXYZ->',bra,conj(bra))
        E1  = einsum('WXYZ,wxYZ,WXwx->',bra,conj(bra),ham[0][0][0])
        E1 += einsum('WXYZ,wXyZ,WYwy->',bra,conj(bra),ham[1][0][0])
        E1 += einsum('WXYZ,WXyz,YZyz->',bra,conj(bra),ham[0][1][0])
        E1 += einsum('WXYZ,WxYz,XZxz->',bra,conj(bra),ham[1][1][0])
        # Contract Energy again
        E2 = peps.calc_op(ham,normalize=False)
        self.assertTrue(abs((norm0-norm1)/norm0) < 1e-10)
        print('Passed Norm1')
        self.assertTrue(abs((norm0-E1)/norm0) < 1e-10)
        print('Passed E1')
        print('Norm from calc_norm = {}'.format(norm0))
        print('Norm from exact contraction {}'.format(norm1))
        print('Norm from Energy calc op = {}'.format(E2))
        print('Norm from Energy exact contraction {}'.format(E1))
        print(norm1,E1,norm0,E2,abs((norm0-E2)/norm0))
        self.assertTrue(abs((norm0-E2)/norm0) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

    def test_energy_itf_contraction(self):
        mpiprint(0,'\n'+'='*50+'\nPeps Energy (Ham=ITF) calculation\n'+'-'*50)
        # Create a PEPS
        from cyclopeps.tools.peps_tools import PEPS
        Nx = 2
        Ny = 2
        d = 2
        D = 3
        peps = PEPS(Nx=Nx,Ny=Ny,d=d,D=D,chi=1000)
        # Get the Hamiltonian
        from cyclopeps.ops.itf import return_op
        ham = return_op(Nx,Ny,(1.,2.))
        # Calculate initial norm
        norm0 = peps.calc_norm()
        print('Norm = {}'.format(norm0))
        # Perform the Exact energy calculation:
        bra = einsum('LDWCM,lMXcu,CdYRm,cmZru->WXYZ',peps[0][0],peps[0][1],peps[1][0],peps[1][1])
        norm1 = einsum('WXYZ,WXYZ->',bra,conj(bra))
        E1  = einsum('WXYZ,wxYZ,WXwx->',bra,conj(bra),ham[0][0][0])
        E1 += einsum('WXYZ,wXyZ,WYwy->',bra,conj(bra),ham[1][0][0])
        E1 += einsum('WXYZ,WXyz,YZyz->',bra,conj(bra),ham[0][1][0])
        E1 += einsum('WXYZ,WxYz,XZxz->',bra,conj(bra),ham[1][1][0])
        # Contract Energy again
        E2 = peps.calc_op(ham,normalize=False)
        print('Energy (exact)   = {}'.format(E1))
        print('Energy (routine) = {}'.format(E2))
        self.assertTrue(abs((E2-E1)/E1) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)


if __name__ == "__main__":
    unittest.main()
