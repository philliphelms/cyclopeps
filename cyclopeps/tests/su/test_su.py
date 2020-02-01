import unittest
from cyclopeps.tools.utils import *
import copy
from cyclopeps.tools.peps_tools import PEPS
from cyclopeps.ops.heis import return_op
from cyclopeps.algs.simple_update import *

class test_cal_energy(unittest.TestCase):
    
    def test_robust(self):
        mpiprint(0,'\n'+'='*50+'\nRobust Simple Update Test\n'+'-'*50)
        # Create a PEPS
        Nx = 5
        Ny = 5
        d = 2
        D = 2
        chi = 10
        # Create a random peps
        peps = PEPS(Nx=Nx,
                    Ny=Ny,
                    d=d,
                    D=D,
                    chi=chi,
                    canonical=True,
                    normalize=False)
        # ----------------------------------------------
        # Check how normalization affects tensors
        peps2 = peps.copy()
        peps2.normalize()
        # lambda tensors comparison
        for i in range(len(peps2.ltensors)):
            for j in range(len(peps2.ltensors[i])):
                for k in range(len(peps2.ltensors[i][j])):
                    assert(np.allclose(peps2.ltensors[i][j][k]/peps.ltensors[i][j][k],1.))
        # Normal tensors comparison
        for i in range(len(peps2.tensors)):
            for j in range(len(peps2.tensors[i])):
                assert(np.allclose(peps2.tensors[i][j],peps.tensors[i][j]))

        # ----------------------------------------------
        # Create a heisenberg operator
        op = return_op(Nx,Ny)

        # ----------------------------------------------
        # Do imaginary time evolution
        res = run_tebd(Nx,Ny,d,op,peps=peps,step_size=0.0,n_step=10,conv_tol=1e-100)

    def test_energy_calc(self):
        from cyclopeps.tools.peps_tools import PEPS
        from cyclopeps.ops.itf import return_op
        from cyclopeps.algs.simple_update import run_tebd
        mpiprint(0,'\n'+'='*50+'\nPeps Normalization test\n'+'-'*50)
        # Create PEPS
        Nx = 5
        Ny = 5
        d = 2
        D = 2
        chi = 10
        # Get mpo
        ham = return_op(Nx,Ny,(1.,2.))
        # Run TEBD
        Ef,_ = run_tebd(Nx,Ny,d,ham,D=D,chi=chi,n_step=100)
        print('Final  E = {}'.format(Ef))

        mpiprint(0,'Passed\n'+'='*50)

if __name__ == "__main__":
    unittest.main()
