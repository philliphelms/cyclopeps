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
        peps = PEPS(Nx=Nx,Ny=Ny,d=d,D=D,chi=chi)
        # Get ops
        ham = return_op(Nx,Ny,(1.,2.))
        norm = peps.calc_norm(chi=20) 
        E = peps.calc_op(ham)
        mpiprint(0,'E = {}'.format(E/norm))
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
        mpiprint(0,'Norm = {}'.format(norm0))
        # Perform the Exact energy calculation:
        bra = einsum('LDWCM,lMXcu,CdYRm,cmZru->WXYZ',peps[0][0],peps[0][1],peps[1][0],peps[1][1])
        norm1 = einsum('WXYZ,WXYZ->',bra,conj(bra))
        E1  = einsum('WXYZ,wxYZ,WXwx->',bra,conj(bra),ham[0][0][0])
        E1 += einsum('WXYZ,wXyZ,WYwy->',bra,conj(bra),ham[1][0][0])
        E1 += einsum('WXYZ,WXyz,YZyz->',bra,conj(bra),ham[0][1][0])
        E1 += einsum('WXYZ,WxYz,XZxz->',bra,conj(bra),ham[1][1][0])
        # Contract Energy again
        E2 = peps.calc_op(ham,normalize=False)
        mpiprint(0,'Energy = {}'.format(E2))
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
        mpiprint(0,'Norm = {}'.format(norm0))
        # Perform the Exact energy calculation:
        bra = einsum('LDWCM,lMXcu,CdYRm,cmZru->WXYZ',peps[0][0],peps[0][1],peps[1][0],peps[1][1])
        norm1 = einsum('WXYZ,WXYZ->',bra,conj(bra))
        E1  = einsum('WXYZ,wxYZ,WXwx->',bra,conj(bra),ham[0][0][0])
        E1 += einsum('WXYZ,wXyZ,WYwy->',bra,conj(bra),ham[1][0][0])
        E1 += einsum('WXYZ,WXyz,YZyz->',bra,conj(bra),ham[0][1][0])
        E1 += einsum('WXYZ,WxYz,XZxz->',bra,conj(bra),ham[1][1][0])
        # Contract Energy again
        E2 = peps.calc_op(ham,normalize=False)
        mpiprint(0,'Energy (exact)   = {}'.format(E1))
        mpiprint(0,'Energy (routine) = {}'.format(E2))
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
        mpiprint(0,'Passed Norm1')
        self.assertTrue(abs((norm0-E1)/norm0) < 1e-10)
        mpiprint(0,'Passed E1')
        mpiprint(0,'Norm from calc_norm = {}'.format(norm0))
        mpiprint(0,'Norm from exact contraction {}'.format(norm1))
        mpiprint(0,'Norm from Energy calc op = {}'.format(E2))
        mpiprint(0,'Norm from Energy exact contraction {}'.format(E1))
        #mpiprint(0,norm1,E1,norm0,E2,abs((norm0-E2)/norm0))
        self.assertTrue(abs((norm0-E2)/norm0) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

    """
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
        mpiprint(0,'Norm = {}'.format(norm0))
        # Perform the Exact energy calculation:
        bra = einsum('abAcd,edBfg,ckDlm,fmEno->ABDE',peps[0][0],peps[0][1],peps[1][0],peps[1][1])
        norm = einsum('ABDE,ABDE->',bra,conj(bra))
        mpiprint(0,'Full contract norm = {}'.format(norm))
        E1_ = einsum('ABDE,abDE,ABab->',bra,conj(bra),ham[0][0][0])
        E2_ = einsum('ABDE,ABde,DEde->',bra,conj(bra),ham[0][1][0])
        E3_ = einsum('ABDE,aBdE,ADad->',bra,conj(bra),ham[1][0][0])
        E4_ = einsum('ABDE,AbDe,BEbe->',bra,conj(bra),ham[1][1][0])
        #mpiprint(0,E1_,E2_,E3_,E4_)
        bra = einsum('LDWCM,lMXcu,CdYRm,cmZru->WXYZ',peps[0][0],peps[0][1],peps[1][0],peps[1][1])
        norm1 = einsum('WXYZ,WXYZ->',bra,conj(bra))
        E1  = einsum('WXYZ,wxYZ,WXwx->',bra,conj(bra),ham[0][0][0])
        E2  = einsum('WXYZ,WXyz,YZyz->',bra,conj(bra),ham[0][1][0])
        E3  = einsum('WXYZ,wXyZ,WYwy->',bra,conj(bra),ham[1][0][0])
        E4  = einsum('WXYZ,WxYz,XZxz->',bra,conj(bra),ham[1][1][0])
        #mpiprint(0,E1,E2,E3,E4)
        E1 = E1+E2+E3+E4
        # Contract Energy again
        E2 = peps.calc_op(ham,normalize=False)
        mpiprint(0,'Energy (exact)   = {}'.format(E1))
        mpiprint(0,'Energy (routine) = {}'.format(E2))
        self.assertTrue(abs((E2-E1)/E1) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)
    """

    def test_energy_itf_contraction_3x3(self):
        mpiprint(0,'\n'+'='*50+'\nPeps Energy (Ham=ITF, 3x3) calculation\n'+'-'*50)
        # Create a PEPS
        from cyclopeps.tools.peps_tools import PEPS
        Nx = 3
        Ny = 3
        d = 2
        D = 3
        peps = PEPS(Nx=Nx,Ny=Ny,d=d,D=D,chi=1000)
        # Get the Hamiltonian
        from cyclopeps.ops.itf import return_op
        ham = return_op(Nx,Ny,(1.,2.))
        # Calculate initial norm
        norm0 = peps.calc_norm()
        mpiprint(0,'Norm = {}'.format(norm0))
        # Perform the Exact energy calculation:
        row1 = einsum('abAcd,edBfg,hgCij->ABCcfi',peps[0][0],peps[0][1],peps[0][2])
        row2 = einsum('ckDlm,fmEno,ioFpq->cfiDEFlnp',peps[1][0],peps[1][1],peps[1][2])
        row3 = einsum('lrGst,ntHuv,pvIwx->lnpGHI',peps[2][0],peps[2][1],peps[2][2])
        bra = einsum('ABCcfi,cfiDEFlnp,lnpGHI->ABCDEFGHI',row1,row2,row3)
        E1 = einsum('ABCDEFGHI,abCDEFGHI,ABab->',bra,conj(bra),ham[0][0][0])
        E2 = einsum('ABCDEFGHI,AbcDEFGHI,BCbc->',bra,conj(bra),ham[0][0][1])
        E3 = einsum('ABCDEFGHI,ABCdeFGHI,DEde->',bra,conj(bra),ham[0][1][0])
        E4 = einsum('ABCDEFGHI,ABCDefGHI,EFef->',bra,conj(bra),ham[0][1][1])
        E5 = einsum('ABCDEFGHI,ABCDEFghI,GHgh->',bra,conj(bra),ham[0][2][0])
        E6 = einsum('ABCDEFGHI,ABCDEFGhi,HIhi->',bra,conj(bra),ham[0][2][1])
        #mpiprint(0,E1,E2,E3,E4,E5,E6)
        E7 = einsum('ABCDEFGHI,aBCdEFGHI,ADad->',bra,conj(bra),ham[1][0][0])
        E8 = einsum('ABCDEFGHI,ABCdEFgHI,DGdg->',bra,conj(bra),ham[1][0][1])
        E9 = einsum('ABCDEFGHI,AbCDeFGHI,BEbe->',bra,conj(bra),ham[1][1][0])
        E10= einsum('ABCDEFGHI,ABCDeFGhI,EHeh->',bra,conj(bra),ham[1][1][1])
        E11= einsum('ABCDEFGHI,ABcDEfGHI,CFcf->',bra,conj(bra),ham[1][2][0])
        E12= einsum('ABCDEFGHI,ABCDEfGHi,FIfi->',bra,conj(bra),ham[1][2][1])
        #mpiprint(0,E7,E8,E9,E10,E11,E12)
        E = E1+E2+E3+E4+E5+E6+E7+E8+E9+E10+E11+E12
        # Contract Energy again
        E2 = peps.calc_op(ham,normalize=False)
        mpiprint(0,'Energy (exact)   = {}'.format(E))
        mpiprint(0,'Energy (routine) = {}'.format(E2))
        self.assertTrue(abs((E2-E)/E) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

if __name__ == "__main__":
    unittest.main()
