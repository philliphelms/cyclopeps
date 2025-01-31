import unittest
from cyclopeps.tools.utils import *
import copy
from cyclopeps.tools.gen_ten import ones,einsum

class test_cal_energy(unittest.TestCase):

    def test_energy_contraction_ones_z2(self):
        mpiprint(0,'\n'+'='*50+'\nPeps Energy (Ham=Identity, peps=ones, Z2 symmetry) calculation\n'+'-'*50)
        # Create a PEPS
        from cyclopeps.tools.peps_tools import PEPS
        Nx = 2
        Ny = 2
        d = 2
        D = 3
        Zn = 2
        backend='numpy'
        peps = PEPS(Nx=Nx,
                    Ny=Ny,
                    d=d,
                    D=D,
                    chi=1000,
                    Zn=2,
                    backend=backend,
                    normalize=False)
        # Set all tensor values to 1
        for xind in range(Nx):
            for yind in range(Ny):
                peps[xind][yind].fill_all(1.)
        # Get the Hamiltonian
        from cyclopeps.ops.identity import return_op
        ham = return_op(Nx,Ny,sym='Z2',backend=backend)
        # Calculate initial norm
        norm0 = peps.calc_norm()*4.
        mpiprint(0,'Norm (routine) = {}'.format(norm0))
        # Perform the Exact energy calculation:
        bra = einsum('LDWCM,lMXcu->LDluWXCc',peps[0][0],peps[0][1]).remove_empty_ind(0).remove_empty_ind(0).remove_empty_ind(0).remove_empty_ind(0)
        bra = einsum('WXCc,CdYRm->dRWXYcm',bra,peps[1][0]).remove_empty_ind(0).remove_empty_ind(0)
        bra = einsum('WXYcm,cmZru->ruWXYZ',bra,peps[1][1]).remove_empty_ind(0).remove_empty_ind(0)
        norm1 = einsum('WXYZ,WXYZ->',bra,bra.conj())
        norm1 = norm1*4.
        mpiprint(0,'Norm (explicit) = {}'.format(norm1))
        tmp = einsum('WXYZ,wxYZ->WXwx',bra,bra.conj())
        E1  = einsum('WXwx,WXwx->',tmp,ham[0][0][0])
        tmp = einsum('WXYZ,wXyZ->WYwy',bra,bra.conj())
        E1 += einsum('WYwy,WYwy->',tmp,ham[1][0][0])
        tmp = einsum('WXYZ,WXyz->YZyz',bra,bra.conj())
        E1 += einsum('YZyz,YZyz->',tmp,ham[0][1][0])
        tmp = einsum('WXYZ,WxYz->XZxz',bra,bra.conj())
        E1 += einsum('XZxz,XZxz->',tmp,ham[1][1][0])
        mpiprint(0,'Explicitly computed energy (not normalized) = {}'.format(E1))
        # Contract Energy again
        E2 = peps.calc_op(ham,normalize=False)
        mpiprint(0,'Energy via peps Method (not normalized) = {}'.format(E2))
        self.assertTrue(abs((norm0-norm1)/norm0) < 1e-10)
        print('Check here {}, {}, {}, {}'.format(norm0,norm1,E1,E2))
        self.assertTrue(abs((norm0-E1)/norm0) < 1e-10)
        self.assertTrue(abs((norm0-E2)/norm0) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

    def test_energy_contraction_heis_z2(self):
        mpiprint(0,'\n'+'='*50+'\nPeps Energy (Ham=Heisenberg, peps=random, Z2 symmetry) calculation\n'+'-'*50)
        # Create a PEPS
        from cyclopeps.tools.peps_tools import PEPS
        Nx = 2
        Ny = 2
        d = 2
        D = 3
        Zn = 2
        backend='numpy'
        peps = PEPS(Nx=Nx,
                    Ny=Ny,
                    d=d,
                    D=D,
                    chi=1000,
                    Zn=2,
                    backend=backend,
                    normalize=False)
        # Get the Hamiltonian
        from cyclopeps.ops.heis import return_op
        ham = return_op(Nx,Ny,sym='Z2',backend=backend)
        # Calculate initial norm
        norm0 = peps.calc_norm()
        mpiprint(0,'Norm (routine) = {}'.format(norm0))
        # Perform the Exact energy calculation:
        bra = einsum('LDWCM,lMXcu->LDluWXCc',peps[0][0],peps[0][1]).remove_empty_ind(0).remove_empty_ind(0).remove_empty_ind(0).remove_empty_ind(0)
        bra = einsum('WXCc,CdYRm->dRWXYcm',bra,peps[1][0]).remove_empty_ind(0).remove_empty_ind(0)
        bra = einsum('WXYcm,cmZru->ruWXYZ',bra,peps[1][1]).remove_empty_ind(0).remove_empty_ind(0)
        norm1 = einsum('WXYZ,WXYZ->',bra,bra.conj())
        norm1 = norm1
        mpiprint(0,'Norm (explicit) = {}'.format(norm1))
        #print(ham[0][0][0])
        tmp = einsum('WXYZ,wxYZ->WXwx',bra,bra.conj())
        E1  = einsum('WXwx,WXwx->',tmp,ham[0][0][0])
        tmp = einsum('WXYZ,wXyZ->WYwy',bra,bra.conj())
        E1 += einsum('WYwy,WYwy->',tmp,ham[1][0][0])
        tmp = einsum('WXYZ,WXyz->YZyz',bra,bra.conj())
        E1 += einsum('YZyz,YZyz->',tmp,ham[0][1][0])
        tmp = einsum('WXYZ,WxYz->XZxz',bra,bra.conj())
        E1 += einsum('XZxz,XZxz->',tmp,ham[1][1][0])
        E1 = E1
        mpiprint(0,'Explicitly computed energy (not normalized) = {}'.format(E1))
        # Contract Energy again
        E2 = peps.calc_op(ham,normalize=False)
        mpiprint(0,'Energy via peps Method (not normalized) = {}'.format(E2))
        self.assertTrue(abs((norm0-norm1)/norm0) < 1e-10)
        self.assertTrue(abs((E2-E1)/E1) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

    def test_energy_contraction_z2(self):
        mpiprint(0,'\n'+'='*50+'\nPeps Energy (Ham=Identity, peps=random, Z2 symmetry) calculation\n'+'-'*50)
        # Create a PEPS
        from cyclopeps.tools.peps_tools import PEPS
        Nx = 2
        Ny = 2
        d = 2
        D = 3
        Zn = 2
        backend='numpy'
        peps = PEPS(Nx=Nx,
                    Ny=Ny,
                    d=d,
                    D=D,
                    chi=1000,
                    Zn=2,
                    backend=backend,
                    normalize=False)
        # Get the Hamiltonian
        from cyclopeps.ops.identity import return_op
        ham = return_op(Nx,Ny,sym='Z2',backend=backend)
        # Calculate initial norm
        norm0 = peps.calc_norm()*4.
        mpiprint(0,'Norm (routine) = {}'.format(norm0))
        # Perform the Exact energy calculation:
        bra = einsum('LDWCM,lMXcu->LDluWXCc',peps[0][0],peps[0][1]).remove_empty_ind(0).remove_empty_ind(0).remove_empty_ind(0).remove_empty_ind(0)
        bra = einsum('WXCc,CdYRm->dRWXYcm',bra,peps[1][0]).remove_empty_ind(0).remove_empty_ind(0)
        bra = einsum('WXYcm,cmZru->ruWXYZ',bra,peps[1][1]).remove_empty_ind(0).remove_empty_ind(0)
        norm1 = einsum('WXYZ,WXYZ->',bra,bra.conj())
        norm1 = norm1*4
        mpiprint(0,'Norm (explicit) = {}'.format(norm1))
        tmp = einsum('WXYZ,wxYZ->WXwx',bra,bra.conj())
        E1  = einsum('WXwx,WXwx->',tmp,ham[0][0][0])
        tmp = einsum('WXYZ,wXyZ->WYwy',bra,bra.conj())
        E1 += einsum('WYwy,WYwy->',tmp,ham[1][0][0])
        tmp = einsum('WXYZ,WXyz->YZyz',bra,bra.conj())
        E1 += einsum('YZyz,YZyz->',tmp,ham[0][1][0])
        tmp = einsum('WXYZ,WxYz->XZxz',bra,bra.conj())
        E1 += einsum('XZxz,XZxz->',tmp,ham[1][1][0])
        mpiprint(0,'Explicitly computed energy (not normalized) = {}'.format(E1))
        # Contract Energy again
        E2 = peps.calc_op(ham,normalize=False)
        mpiprint(0,'Energy via peps Method (not normalized) = {}'.format(E2))
        self.assertTrue(abs((norm0-norm1)/norm0) < 1e-10)
        self.assertTrue(abs((E2-E1)/E2) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

    def test_energy_itf_contraction_ones(self):
        mpiprint(0,'\n'+'='*50+'\nPeps Energy (Ham=ITF, peps=ones, no symmetry) calculation\n'+'-'*50)
        # Create a PEPS
        from cyclopeps.tools.peps_tools import PEPS
        Nx = 2
        Ny = 2
        d = 2
        D = 3
        peps = PEPS(Nx=Nx,Ny=Ny,d=d,D=D,chi=1000,normalize=False)
        # Set all tensor values to 1
        for xind in range(Nx):
            for yind in range(Ny):
                peps[xind][yind].fill_all(1.)
        # Get the Hamiltonian
        from cyclopeps.ops.itf import return_op
        ham = return_op(Nx,Ny,(1.,2.))
        # Calculate initial norm
        norm0 = peps.calc_norm()
        mpiprint(0,'Norm = {}'.format(norm0))
        # Perform the Exact energy calculation:
        bra = einsum('LDWCM,lMXcu->WXCc',peps[0][0],peps[0][1])
        bra = einsum('WXCc,CdYRm->WXYcm',bra,peps[1][0])
        bra = einsum('WXYcm,cmZru->WXYZ',bra,peps[1][1])
        norm1 = einsum('WXYZ,WXYZ->',bra,bra.conj())
        mpiprint(0,'Explicitly computed norm = {}'.format(norm1))
        tmp = einsum('WXYZ,wxYZ->WXwx',bra,bra.conj())
        E1  = einsum('WXwx,WXwx->',tmp,ham[0][0][0])
        tmp = einsum('WXYZ,wXyZ->WYwy',bra,bra.conj())
        E1 += einsum('WYwy,WYwy->',tmp,ham[1][0][0])
        tmp = einsum('WXYZ,WXyz->YZyz',bra,bra.conj())
        E1 += einsum('YZyz,YZyz->',tmp,ham[0][1][0])
        tmp = einsum('WXYZ,WxYz->XZxz',bra,bra.conj())
        E1 += einsum('XZxz,XZxz->',tmp,ham[1][1][0])
        mpiprint(0,'Explicitly computed energy (not normalized) = {}'.format(E1))
        # Contract Energy again
        E2 = peps.calc_op(ham,normalize=False)
        mpiprint(0,'Energy (routine) = {}'.format(E2))
        self.assertTrue(abs((E2-E1)/E1) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

    def test_energy_contraction_ones(self):
        mpiprint(0,'\n'+'='*50+'\nPeps Energy (Ham=Identity, peps=ones, no symmetry) calculation\n'+'-'*50)
        # Create a PEPS
        from cyclopeps.tools.peps_tools import PEPS
        Nx = 2
        Ny = 2
        d = 2
        D = 3
        peps = PEPS(Nx=Nx,
                    Ny=Ny,
                    d=d,
                    D=D,
                    chi=1000,
                    Zn=None,
                    backend='numpy',
                    normalize=False)
        # Set all tensor values to 1
        for xind in range(Nx):
            for yind in range(Ny):
                peps[xind][yind].fill_all(1.)
        # Get the Hamiltonian
        from cyclopeps.ops.identity import return_op
        ham = return_op(Nx,Ny)
        # Calculate initial norm
        norm0 = peps.calc_norm()*4.
        mpiprint(0,'Norm = {}'.format(norm0))
        # Perform the Exact energy calculation:
        bra = einsum('LDWCM,lMXcu->WXCc',peps[0][0],peps[0][1])
        bra = einsum('WXCc,CdYRm->WXYcm',bra,peps[1][0])
        bra = einsum('WXYcm,cmZru->WXYZ',bra,peps[1][1])
        norm1 = einsum('WXYZ,WXYZ->',bra,bra.conj())*4.
        tmp = einsum('WXYZ,wxYZ->WXwx',bra,bra.conj())
        E1  = einsum('WXwx,WXwx->',tmp,ham[0][0][0])
        tmp = einsum('WXYZ,wXyZ->WYwy',bra,bra.conj())
        E1 += einsum('WYwy,WYwy->',tmp,ham[1][0][0])
        tmp = einsum('WXYZ,WXyz->YZyz',bra,bra.conj())
        E1 += einsum('YZyz,YZyz->',tmp,ham[0][1][0])
        tmp = einsum('WXYZ,WxYz->XZxz',bra,bra.conj())
        E1 += einsum('XZxz,XZxz->',tmp,ham[1][1][0])
        mpiprint(0,'Explicitly computed energy (not normalized) = {}'.format(E1))
        # Contract Energy again
        E2 = peps.calc_op(ham,normalize=False)
        mpiprint(0,'Energy via peps Method (not normalized) = {}'.format(E2))
        self.assertTrue(abs((norm0-norm1)/norm0) < 1e-10)
        self.assertTrue(abs((norm0-E1)/norm0) < 1e-10)
        self.assertTrue(abs((norm0-E2)/norm0) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

    def test_energy_contraction(self):
        mpiprint(0,'\n'+'='*50+'\nPeps Energy (Ham=Identity, peps=random, no symmetry) calculation\n'+'-'*50)
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
        norm0 = peps.calc_norm()*4.
        # Perform the Exact energy calculation:
        bra = einsum('LDWCM,lMXcu->WXCc',peps[0][0],peps[0][1])
        bra = einsum('WXCc,CdYRm->WXYcm',bra,peps[1][0])
        bra = einsum('WXYcm,cmZru->WXYZ',bra,peps[1][1])
        norm1 = einsum('WXYZ,WXYZ->',bra,bra.conj())*4.
        tmp = einsum('WXYZ,wxYZ->WXwx',bra,bra.conj())
        E1  = einsum('WXwx,WXwx->',tmp,ham[0][0][0])
        tmp = einsum('WXYZ,wXyZ->WYwy',bra,bra.conj())
        E1 += einsum('WYwy,WYwy->',tmp,ham[1][0][0])
        tmp = einsum('WXYZ,WXyz->YZyz',bra,bra.conj())
        E1 += einsum('YZyz,YZyz->',tmp,ham[0][1][0])
        tmp = einsum('WXYZ,WxYz->XZxz',bra,bra.conj())
        E1 += einsum('XZxz,XZxz->',tmp,ham[1][1][0])
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

    def test_energy_itf_contraction(self):
        mpiprint(0,'\n'+'='*50+'\nPeps Energy (Ham=ITF, peps=random, no symmetry) calculation\n'+'-'*50)
        # Create a PEPS
        from cyclopeps.tools.peps_tools import PEPS
        Nx = 2
        Ny = 2
        d = 2
        D = 3
        peps = PEPS(Nx=Nx,Ny=Ny,d=d,D=D,chi=1000)
        peps2= PEPS(Nx=Nx,Ny=Ny,d=d,D=D,chi=1000)
        # Get the Hamiltonian
        from cyclopeps.ops.itf import return_op
        ham = return_op(Nx,Ny,(1.,2.))
        # Perform the Exact energy calculation:
        bra = einsum('LDWCM,lMXcu->WXCc',peps[0][0],peps[0][1])
        bra = einsum('WXCc,CdYRm->WXYcm',bra,peps[1][0])
        bra = einsum('WXYcm,cmZru->WXYZ',bra,peps[1][1])
        ket = einsum('LDWCM,lMXcu->WXCc',peps2[0][0],peps2[0][1])
        ket = einsum('WXCc,CdYRm->WXYcm',ket,peps2[1][0])
        ket = einsum('WXYcm,cmZru->WXYZ',ket,peps2[1][1])
        norm1 = einsum('WXYZ,WXYZ->',bra,ket)
        tmp = einsum('WXYZ,wxYZ->WXwx',bra,ket)
        E1 = einsum('WXwx,WXwx->',tmp,ham[0][0][0])
        tmp = einsum('WXYZ,wXyZ->WYwy',bra,ket)
        E2 = einsum('WYwy,WYwy->',tmp,ham[1][0][0])
        tmp = einsum('WXYZ,WXyz->YZyz',bra,ket)
        E3 = einsum('YZyz,YZyz->',tmp,ham[0][1][0])
        tmp = einsum('WXYZ,WxYz->XZxz',bra,ket)
        E4 = einsum('XZxz,XZxz->',tmp,ham[1][1][0])
        E1 = E1+E2+E3+E4
        # Contract Energy again
        E2 = peps.calc_op(ham,normalize=False,chi=1e100,ket=peps2)
        mpiprint(0,'Energy (exact)   = {}'.format(E1))
        mpiprint(0,'Energy (routine) = {}'.format(E2))
        self.assertTrue(abs((E2-E1)/E1) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

if __name__ == "__main__":
    unittest.main()
