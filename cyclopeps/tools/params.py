import os, sys
from mpi4py import MPI

# MPI Global Variables
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.size

# Temporary directories for calculation
TMPDIR = os.environ.get('TMPDIR','.')
TMPDIR = os.environ.get('CYCLOPEPS_TMPDIR',TMPDIR)

# Printing Global Variables
DEBUG = False
VERBOSE = 1
VERBOSE_TIME = 3
VERBOSE_MEM = 2
OUTPUT_DIGITS = 5
OUTPUT_COLS = 5

# Eigenproblem parameters
DAVIDSON_TOL = 1e-16
DAVIDSON_MAX_ITER = 100
USE_PRECOND = False
ARNOLDI_TOL = 1e-8
ARNOLDI_MAX_ITER = 100

# Memory Global Variables
import psutil
_,av,_,_,_,_,_,_,_,_,_ = psutil.virtual_memory()
MAX_MEMORY = av
