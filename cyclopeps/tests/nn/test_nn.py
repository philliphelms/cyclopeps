import unittest
from cyclopeps.tools.utils import *
import copy

class test_peps(unittest.TestCase):

    def test_small_nn_energy_calc(self):
        mpiprint(0,'\n'+'='*50+'\nNext Nearest Neighbor Energy calc (2x2)\n'+'-'*50)
        from cyclopeps.tools.peps_tools import PEPS
        from cyclopeps.ops.j1j2 import return_op
        from cyclopeps.ops.heis import return_op as return_heis
        # Get inputs
        Nx = 2
        Ny = 2
        D  = 4
        chi= 10000
        backend = 'numpy'
        d = 2

        # Get mpo
        ham = return_op(Nx,Ny,sym=None,backend=backend)
        heisham = return_heis(Nx,Ny,sym=None,backend=backend)

        # Create the peps
        peps = PEPS(Nx=Nx,
                    Ny=Ny,
                    d=d,
                    D=D,
                    chi=chi,
                    backend=backend)

        # Calculate the operator
        E1 = peps.calc_op(ham,chi=chi,nn=True,contracted_env=True)
        mpiprint(0,'Contracted environment next nearest: {}'.format(E1))
        E2 = peps.calc_op(ham,chi=chi,nn=True,contracted_env=False)
        mpiprint(0,'bound mpo  environment next nearest: {}'.format(E2))
        E3 = peps.calc_op(heisham)
        mpiprint(0,'Only nearest neighbor (standard):    {}'.format(E3))
        self.assertTrue(abs((E1-E3)/E3) < 1e-3)
        self.assertTrue(abs((E2-E3)/E3) < 1e-3)
        mpiprint(0,'Passed\n'+'='*50)

    def test_large_nn_energy_calc(self):
        mpiprint(0,'\n'+'='*50+'\nNext Nearest Neighbor Energy calc (5x5)\n'+'-'*50)
        from cyclopeps.tools.peps_tools import PEPS
        from cyclopeps.ops.j1j2 import return_op
        from cyclopeps.ops.heis import return_op as return_heis
        # Get inputs
        Nx = 5
        Ny = 5
        D  = 2
        chi= 100
        backend = 'numpy'
        d = 2

        # Get mpo
        ham = return_op(Nx,Ny,sym=None,backend=backend)
        heisham = return_heis(Nx,Ny,sym=None,backend=backend)

        # Create the peps
        peps = PEPS(Nx=Nx,
                    Ny=Ny,
                    d=d,
                    D=D,
                    chi=chi,
                    backend=backend)

        # Calculate the operator
        E1 = peps.calc_op(ham,chi=chi,nn=True,contracted_env=True)
        mpiprint(0,'Contracted environment next nearest: {}'.format(E1))
        E2 = peps.calc_op(ham,chi=chi,nn=True,contracted_env=False)
        mpiprint(0,'bound mpo  environment next nearest: {}'.format(E2))
        E3 = peps.calc_op(heisham)
        mpiprint(0,'Only nearest neighbor (standard):    {}'.format(E3))
        self.assertTrue(abs((E1-E3)/E3) < 1e-3)
        self.assertTrue(abs((E2-E3)/E3) < 1e-3)
        mpiprint(0,'Passed\n'+'='*50)

    def test_fu(self):
        mpiprint(0,'\n'+'='*50+'\nNext Nearest Neighbor full update (5x5)\n'+'-'*50)
        from cyclopeps.ops.j1j2 import return_op
        from cyclopeps.algs.nntebd import run_tebd as fu
        Nx = 5
        Ny = 5
        D = 2
        d = 2
        chi = 10
        backend='numpy'

        step_sizes = [0.1,0.01]
        n_step = [5,5]

        # Get mpos 
        ham = return_op(Nx,Ny,sym=None,backend=backend)

        # Run SU/FU
        Ef,_ = fu(Nx,
                  Ny,
                  d,
                  ham,
                  D=D,
                  chi=chi,
                  backend=backend,
                  n_step=n_step,
                  step_size=step_sizes)

        print('Final  E = {}'.format(Ef))
        mpiprint(0,'Passed\n'+'='*50)


if __name__ == "__main__":
    unittest.main()
