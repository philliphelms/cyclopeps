import unittest
from cyclopeps.tools.utils import *
import copy
from cyclopeps.tools.peps_tools import PEPS
from cyclopeps.ops.heis import return_op
from cyclopeps.algs.simple_update import *

class test_cal_energy(unittest.TestCase):

    def test_heis_z2(self):
        from cyclopeps.tools.peps_tools import PEPS
        from cyclopeps.ops.heis import return_op
        from cyclopeps.algs.simple_update import run_tebd,tebd_step
        mpiprint(0,'\n'+'='*50+'\nSU Z2 Heisenberg test\n'+'-'*50)
        # Create PEPS
        Nx = 5
        Ny = 5
        d = 2
        D = 4
        chi = 10
        Zn = 2
        backend = 'numpy'
        # Get operator
        ham = return_op(Nx,Ny,sym="Z2",backend=backend)
        # Run TEBD
        Ef,_ = run_tebd(Nx,
                        Ny,
                        d,
                        ham,
                        D=D,
                        chi=chi,
                        Zn=Zn,
                        backend=backend,
                        step_size=[0.5,0.1,0.05,0.01],
                        n_step=1)
        print('Final  E = {}'.format(Ef))
        mpiprint(0,'Passed\n'+'='*50)

    def test_heis(self):
        from cyclopeps.tools.peps_tools import PEPS
        from cyclopeps.ops.heis import return_op
        from cyclopeps.algs.simple_update import run_tebd,tebd_step
        mpiprint(0,'\n'+'='*50+'\nSU Heisenberg test\n'+'-'*50)
        # Create PEPS
        Nx = 5
        Ny = 5
        d = 2
        D = 2
        chi = 10
        Zn = None
        backend = 'numpy'
        # Get operator
        ham = return_op(Nx,Ny,sym=None,backend=backend)
        # Run TEBD
        Ef,_ = run_tebd(Nx,
                        Ny,
                        d,
                        ham,
                        D=D,
                        chi=chi,
                        Zn=Zn,
                        backend=backend,
                        step_size=[0.5,0.1,0.05,0.01],
                        n_step=1)
        print('Final  E = {}'.format(Ef))
        mpiprint(0,'Passed\n'+'='*50)

if __name__ == "__main__":
    unittest.main()
