from cyclopeps.tools.utils import *
from cyclopeps.tools.peps_tools import PEPS
from cyclopeps.ops.heis import return_op
from cyclopeps.algs.simple_update import run_tebd,tebd_step
import cProfile,pstats

# Create PEPS
Nx = 10
Ny = 10
d = 2
D = 2
chi = 10
Zn = 2
backend = 'numpy'

# Get operator
ham = return_op(Nx,Ny,sym="Z2",backend=backend)

# Set up profiler
pr = cProfile.Profile()

# Start Profiling
pr.enable()

# Run TEBD
Ef,_ = run_tebd(Nx,
                Ny,
                d,
                ham,
                D=D,
                chi=chi,
                Zn=Zn,
                backend=backend,
                step_size=[0.1],
                n_step=10)

# End profiling 
pr.disable()

# Print Results
ps = pstats.Stats(pr).sort_stats('cumulative')
ps.print_stats()
