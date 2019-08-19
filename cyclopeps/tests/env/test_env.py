import unittest
from cyclopeps.tools.utils import *
import copy

class test_env(unittest.TestCase):

    def test_small(self):
        mpiprint(0,'\n'+'='*50+'\nTesting d=2, D=3, (Nx,Ny) = (2,2)\n'+'-'*50)
        from cyclopeps.tools.peps_tools import make_rand_peps, calc_peps_norm
        # Calculation Parameters
        Nx = 2
        Ny = 2
        d = 2
        D = 3
        # Make a random PEPS
        peps = make_rand_peps(Nx,Ny,d,D)
        # Compute the norm exactly
        bra = einsum('abPcd,edQfg,chRij,fjSkl->PQRS',peps[0][0],peps[0][1],peps[1][0],peps[1][1])
        norm = einsum('PQRS,PQRS->',bra,conj(bra))
        # Compute the norm exactly using Boundary MPO
        norm2 = calc_peps_norm(peps,chi=100)
        self.assertTrue(abs(norm-norm2)/abs(norm) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)


    def test_d1D1N2(self):
        mpiprint(0,'\n'+'='*50+'\nTesting d=1, D=1, (Nx,Ny) = (2,2)\n'+'-'*50)
        from cyclopeps.tools.peps_tools import make_rand_peps, calc_peps_norm
        # Calculation Parameters
        Nx = 2
        Ny = 2
        d = 1
        D = 1
        # Make a random PEPS
        peps = make_rand_peps(Nx,Ny,d,D)
        pepscopy = copy.deepcopy(peps)
        for x in range(len(peps)):
            for y in range(len(peps[0])):
                peps[x][y] = einsum('ldpru->p',peps[x][y])
        # Compute the norm exactly
        bra = einsum('a,b,c,d->abcd',
                        peps[0][0],peps[0][1],
                        peps[1][0],peps[1][1])
        norm1 = einsum('abcd,abcd->',bra,conj(bra))
        # Compute the norm exactly using Boundary MPO
        norm2 = calc_peps_norm(pepscopy,chi=100)
        # Check that they are similar
        self.assertTrue(abs(norm1-norm2)/abs(norm1) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

    def test_d1D1N5(self):
        mpiprint(0,'\n'+'='*50+'\nTesting d=1, D=1, (Nx,Ny) = (5,5)\n'+'-'*50)
        from cyclopeps.tools.peps_tools import make_rand_peps, calc_peps_norm
        # Calculation Parameters
        Nx = 5
        Ny = 5
        d = 1
        D = 1
        # Make a random PEPS
        peps = make_rand_peps(Nx,Ny,d,D)
        pepscopy = copy.deepcopy(peps)
        for x in range(len(peps)):
            for y in range(len(peps[0])):
                peps[x][y] = einsum('ldpru->p',peps[x][y])
        # Compute the norm exactly
        bra = einsum('a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y->abcdefghijklmnopqrstuvwxy',
                        peps[0][0],peps[0][1],peps[0][2],peps[0][3],peps[0][4],
                        peps[1][0],peps[1][1],peps[1][2],peps[1][3],peps[1][4],
                        peps[2][0],peps[2][1],peps[2][2],peps[2][3],peps[2][4],
                        peps[3][0],peps[3][1],peps[3][2],peps[3][3],peps[3][4],
                        peps[4][0],peps[4][1],peps[4][2],peps[4][3],peps[4][4])
        norm1 = einsum('abcdefghijklmnopqrstuvwxy,abcdefghijklmnopqrstuvwxy->',bra,conj(bra))
        # Compute the norm exactly using Boundary MPO
        norm2 = calc_peps_norm(pepscopy,chi=100)
        # Check that they are similar
        self.assertTrue(abs(norm1-norm2)/abs(norm1) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

    def test_d2D1N5(self):
        mpiprint(0,'\n'+'='*50+'\nTesting d=2, D=1, (Nx,Ny) = (5,5)\n'+'-'*50)
        from cyclopeps.tools.peps_tools import make_rand_peps, calc_peps_norm
        # Calculation Parameters
        Nx = 5
        Ny = 5
        d = 2
        D = 1
        # Make a random PEPS
        peps = make_rand_peps(Nx,Ny,d,D)
        pepscopy = copy.deepcopy(peps)
        for x in range(len(peps)):
            for y in range(len(peps[0])):
                peps[x][y] = einsum('ldpru->p',peps[x][y])
        # Compute the norm exactly
        bra = einsum('a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y->abcdefghijklmnopqrstuvwxy',
                        peps[0][0],peps[0][1],peps[0][2],peps[0][3],peps[0][4],
                        peps[1][0],peps[1][1],peps[1][2],peps[1][3],peps[1][4],
                        peps[2][0],peps[2][1],peps[2][2],peps[2][3],peps[2][4],
                        peps[3][0],peps[3][1],peps[3][2],peps[3][3],peps[3][4],
                        peps[4][0],peps[4][1],peps[4][2],peps[4][3],peps[4][4])
        norm1 = einsum('abcdefghijklmnopqrstuvwxy,abcdefghijklmnopqrstuvwxy->',bra,conj(bra))
        # Compute the norm exactly using Boundary MPO
        norm2 = calc_peps_norm(pepscopy,chi=100)
        # Check that they are similar
        self.assertTrue(abs(norm1-norm2)/abs(norm1) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

    def test_d1D2N2(self):
        mpiprint(0,'\n'+'='*50+'\nTesting d=1, D=2, (Nx,Ny) = (2,2)\n'+'-'*50)

        from cyclopeps.tools.peps_tools import make_rand_peps, calc_peps_norm
        from cyclopeps.tools.env_tools import calc_left_bound_mpo,calc_right_bound_mpo
        # Calculation Parameters
        Nx = 2
        Ny = 2
        d = 1
        D = 2
        # Make a random PEPS
        peps = make_rand_peps(Nx,Ny,d,D)
        # Compute the norm exactly
        bra = einsum('abPcd,edQfg,chRij,fjSkl->PQRS',
                        peps[0][0],peps[0][1],
                        peps[1][0],peps[1][1])
        norm1 = einsum('PQRS,PQRS->',bra,conj(bra))
        # Compute the norm exactly using Boundary MPO
        norm2 = calc_peps_norm(peps,chi=100)
        # Check that they are similar
        self.assertTrue(abs(norm1-norm2)/abs(norm1) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

    def test_left_bmpo(self):
        mpiprint(0,'\n'+'='*50+'\nTesting left boundary mpo\n'+'-'*50)
        from cyclopeps.tools.peps_tools import make_rand_peps, calc_peps_norm
        from cyclopeps.tools.env_tools import calc_left_bound_mpo,calc_right_bound_mpo
        # Calculation Parameters
        Nx = 2
        Ny = 2
        d = 2
        D = 4
        # Make a random PEPS
        peps = make_rand_peps(Nx,Ny,d,D)
        # Compute the left boundary mpo exactly
        bmpo = [None]*2
        bmpo[0] = einsum('ldpru,LDpRU->lLdDrRuU',peps[0][0],peps[0][0])
        bmpo[0] = reshape(bmpo[0],(D,D,D*D))
        bmpo[1] = einsum('ldpru,LDpRU->lLdDrRuU',peps[0][1],peps[0][1])
        bmpo[1] = reshape(bmpo[1],(D*D,D,D))
        # Compute the left boundary mpo with routine
        _bmpo2 = [None]*2
        bmpo2 = calc_left_bound_mpo(peps,Nx,chi=1000,truncate=False)
        _bmpo2[0] = einsum('mrn,nRo->Rro',bmpo2[0],bmpo2[1])
        _bmpo2[1] = einsum('mrn,nRo->mrR',bmpo2[2],bmpo2[3])
        self.assertTrue(abs(summ(bmpo[0]-_bmpo2[0])) < 1e-10)
        self.assertTrue(abs(summ(bmpo[1]-_bmpo2[1])) < 1e-10)
        # Check if full left state is correct
        bmpo = einsum('abPcd,efPgh,idQjk,lhQmn->cgjm',peps[0][0],peps[0][0],peps[0][1],peps[0][1])
        _bmpo2 = einsum('aBc,cDe,eFg,gHi->BDFH',bmpo2[0],bmpo2[1],bmpo2[2],bmpo2[3])
        self.assertTrue(abs(summ(bmpo-_bmpo2)) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)
    
    def test_right_bmpo(self):
        mpiprint(0,'\n'+'='*50+'\nTesting right boundary mpo\n'+'-'*50)
        from cyclopeps.tools.peps_tools import make_rand_peps, calc_peps_norm
        from cyclopeps.tools.env_tools import calc_left_bound_mpo,calc_right_bound_mpo
        # Calculation Parameters
        Nx = 2
        Ny = 2
        d = 2
        D = 4
        # Make a random PEPS
        peps = make_rand_peps(Nx,Ny,d,D)
        # Compute the right boundary mpo exactly
        bmpo = [None]*2
        bmpo[0] = einsum('ldpru,LDpRU->lLdDrRuU',peps[1][0],peps[1][0])
        bmpo[0] = reshape(bmpo[0],(D,D,D*D))
        bmpo[1] = einsum('ldpru,LDpRU->lLdDrRuU',peps[1][1],peps[1][1])
        bmpo[1] = reshape(bmpo[1],(D*D,D,D))
        # Compute the right boundary mpo exactly
        bmpo2 = calc_right_bound_mpo(peps,Nx-2,chi=1000,truncate=False)
        _bmpo2 = [None]*2
        _bmpo2[0] = einsum('mrn,nRo->rRo',bmpo2[0],bmpo2[1])
        _bmpo2[1] = einsum('mrn,nRo->mrR',bmpo2[2],bmpo2[3])
        self.assertTrue(abs(summ(bmpo[0]-_bmpo2[0])) < 1e-10)
        self.assertTrue(abs(summ(bmpo[1]-_bmpo2[1])) < 1e-10)
        # Check if the full state is correct
        bmpo = einsum('abPcd,efPgh,idQjk,lhQmn->aeil',peps[1][0],peps[1][0],peps[1][1],peps[1][1])
        _bmpo2 = einsum('aBc,cDe,eFg,gHi->BDFH',bmpo2[0],bmpo2[1],bmpo2[2],bmpo2[3])
        self.assertTrue(abs(summ(bmpo-_bmpo2)) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

    def test_left_right_bmpo_2x2(self):
        mpiprint(0,'\n'+'='*50+'\nTesting left and right boundary mpo (2x2)\n'+'-'*50)
        from cyclopeps.tools.peps_tools import make_rand_peps, calc_peps_norm
        from cyclopeps.tools.env_tools import calc_left_bound_mpo,calc_right_bound_mpo
        # Calculation Parameters
        Nx = 2
        Ny = 2
        d = 2
        D = 2
        # Make a random PEPS
        peps = make_rand_peps(Nx,Ny,d,D)
        # Compute the norm exactly
        bra = einsum('abPcd,edQfg,chRij,fjSkl->PQRS',
                        peps[0][0],peps[0][1],
                        peps[1][0],peps[1][1])
        norm1 = einsum('PQRS,PQRS->',bra,conj(bra))
        # Compute the right boundary mpo exactly
        lbmpo = einsum('abPcd,efPgh,idQjk,lhQmn->cgjm',peps[0][0],peps[0][0],peps[0][1],peps[0][1])
        rbmpo = einsum('abPcd,efPgh,idQjk,lhQmn->aeil',peps[1][0],peps[1][0],peps[1][1],peps[1][1])
        norm2 = einsum('abcd,badc->',lbmpo,rbmpo)
        # Compute the norm exactly using Boundary MPO
        norm3 = calc_peps_norm(peps,chi=100)
        # Check that these are equal
        self.assertTrue(abs(norm3-norm1)/abs(norm1) < 1e-10)
        self.assertTrue(abs(norm2-norm1)/abs(norm1) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

    def test_left_right_bmpo_3x3(self):
        mpiprint(0,'\n'+'='*50+'\nTesting left and right boundary mpo (3x3)\n'+'-'*50)
        from cyclopeps.tools.peps_tools import make_rand_peps, calc_peps_norm
        from cyclopeps.tools.env_tools import calc_left_bound_mpo,calc_right_bound_mpo
        # Calculation Parameters
        Nx = 3
        Ny = 3
        d = 1
        D = 2
        # Make a random PEPS
        print('making peps')
        peps = make_rand_peps(Nx,Ny,d,D)
        # Compute the norm exactly
        print('Trying to contract a monster')
        row1 = einsum('abAcd,kdDlm,rmGst->AcDlGs',peps[0][0],peps[0][1],peps[0][2])
        row2 = einsum('ceBfg,lgEno,soHuv->cBflEnsHu',peps[1][0],peps[1][1],peps[1][2])
        row3 = einsum('fhCij,njFpq,uqIwx->fCnFuI',peps[2][0],peps[2][1],peps[2][2])
        bra = einsum('AcDlGs,cBflEnsHu,fCnFuI->ABCDEFGHI',row1,row2,row3)
        print('Trying to contract a 2nd monster')
        norm1 = einsum('ABCDEFGHI,ABCDEFGHI->',bra,conj(bra))
        print('Full norm = {}'.format(norm1))
        # Compute the norm exactly using Boundary MPO
        print('Calculating peps norm')
        norm2 = calc_peps_norm(peps,chi=10)
        print('Norm from routine = {}'.format(norm2))
        # Check that these are equal
        self.assertTrue(abs(norm2-norm1)/abs(norm1) < 1e-10)
        mpiprint(0,'Passed\n'+'='*50)

if __name__ == "__main__":
    unittest.main()
