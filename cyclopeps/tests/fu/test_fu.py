import unittest
from cyclopeps.tools.utils import *
import copy

class test_cal_energy(unittest.TestCase):

    def test_energy_calc_heis_Z2_not_reduced(self):
        from cyclopeps.tools.peps_tools import PEPS
        from cyclopeps.ops.heis import return_op
        from cyclopeps.algs.tebd import run_tebd
        mpiprint(0,'\n'+'='*50+'\nPeps Full Update Z2 (not reduced) Heisenberg test\n'+'-'*50)
        # Create PEPS
        Nx = 2
        Ny = 2
        d = 2
        D = 2
        Zn = 2
        reduced=False
        backend = 'numpy'
        chi = 50
        # Get mpo
        ham = return_op(Nx,Ny,sym='Z2',backend=backend)
        # Run TEBD
        Ef,_ = run_tebd(Nx,
                        Ny,
                        d,
                        ham,
                        D=D,
                        Zn=Zn,
                        chi=chi,
                        backend=backend,
                        n_step=100,
                        step_size=0.1,
                        su_step_size=0.1,
                        su_conv_tol=1e-5,
                        reduced=reduced,
                        su_n_step=1)
        print('Final  E = {}'.format(Ef))
        mpiprint(0,'Passed\n'+'='*50)

    def test_energy_calc_heis_not_reduced(self):
        from cyclopeps.tools.peps_tools import PEPS
        from cyclopeps.ops.heis import return_op
        from cyclopeps.algs.tebd import run_tebd
        mpiprint(0,'\n'+'='*50+'\nPeps Full Update (not reduced) Heisenberg test\n'+'-'*50)
        # Create PEPS
        Nx = 4
        Ny = 4
        d = 2
        D = 3
        Zn = None
        backend = 'numpy'
        chi = 50
        # Get mpo
        ham = return_op(Nx,Ny,sym=None,backend=backend)
        # Run TEBD
        Ef,_ = run_tebd(Nx,
                        Ny,
                        d,
                        ham,
                        D=D,
                        Zn=Zn,
                        chi=chi,
                        backend=backend,
                        n_step=100,
                        step_size=0.1,
                        su_step_size=0.1,
                        su_conv_tol=1e-5,
                        reduced=False,
                        su_n_step=1)
        print('Final  E = {}'.format(Ef))
        mpiprint(0,'Passed\n'+'='*50)

    def test_energy_calc_heis(self):
        from cyclopeps.tools.peps_tools import PEPS
        from cyclopeps.ops.heis import return_op
        from cyclopeps.algs.tebd import run_tebd
        mpiprint(0,'\n'+'='*50+'\nPeps Full Update Heisenberg test\n'+'-'*50)
        # Create PEPS
        Nx = 4
        Ny = 4
        d = 2
        D = 3
        Zn = None
        backend = 'numpy'
        chi = 50
        # Get mpo
        ham = return_op(Nx,Ny,sym=None,backend=backend)
        # Run TEBD
        Ef,_ = run_tebd(Nx,
                        Ny,
                        d,
                        ham,
                        D=D,
                        Zn=Zn,
                        chi=chi,
                        backend=backend,
                        n_step=100,
                        su_step_size=0.1,
                        su_conv_tol=1e-5,
                        su_n_step=1)
        print('Final  E = {}'.format(Ef))
        mpiprint(0,'Passed\n'+'='*50)

    def test_energy_calc_heis_Z2_not_reduced(self):
        from cyclopeps.tools.peps_tools import PEPS
        from cyclopeps.ops.heis import return_op
        from cyclopeps.algs.tebd import run_tebd
        mpiprint(0,'\n'+'='*50+'\nPeps Full Update Z2 (not reduced) Heisenberg test\n'+'-'*50)
        # Create PEPS
        Nx = 4
        Ny = 4
        d = 2
        D = 2
        Zn = 2
        backend = 'numpy'
        chi = 50
        # Get mpo
        ham = return_op(Nx,Ny,sym='Z2',backend=backend)
        # Run TEBD
        Ef,_ = run_tebd(Nx,
                        Ny,
                        d,
                        ham,
                        D=D,
                        Zn=Zn,
                        chi=chi,
                        backend=backend,
                        n_step=1,
                        su_step_size=[0.5,0.1],
                        su_conv_tol=1e-5,
                        reduced=False,
                        su_n_step=1)
        print('Final  E = {}'.format(Ef))
        mpiprint(0,'Passed\n'+'='*50)

    def test_energy_calc_heis_Z2(self):
        from cyclopeps.tools.peps_tools import PEPS
        from cyclopeps.ops.heis import return_op
        from cyclopeps.algs.tebd import run_tebd
        mpiprint(0,'\n'+'='*50+'\nPeps Full Update Z2 Heisenberg test\n'+'-'*50)
        # Create PEPS
        Nx = 4
        Ny = 4
        d = 2
        D = 2
        Zn = 2
        backend = 'numpy'
        chi = 50
        # Get mpo
        ham = return_op(Nx,Ny,sym='Z2',backend=backend)
        # Run TEBD
        Ef,_ = run_tebd(Nx,
                        Ny,
                        d,
                        ham,
                        D=D,
                        Zn=Zn,
                        chi=chi,
                        backend=backend,
                        n_step=1,
                        su_step_size=[0.5,0.1],
                        su_conv_tol=1e-5,
                        su_n_step=1)
        print('Final  E = {}'.format(Ef))
        mpiprint(0,'Passed\n'+'='*50)

if __name__ == "__main__":
    unittest.main()
