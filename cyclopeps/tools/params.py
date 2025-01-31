import os, sys
import uuid
try:
    from mpi4py import MPI
    # MPI Global Variables
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.size
except:
    RANK = 0
    SIZE = 1

# Constrict Printing to only rank 0
if RANK != 0:
    sys.stdout = open(os.devnull,'w')
    sys.stderr = open(os.devnull,'w')

# Temporary directories for calculation
TMPDIR = os.environ.get('TMPDIR','.')
TMPDIR = os.environ.get('CYCLOPEPS_TMPDIR',TMPDIR)
DIRID = str(uuid.uuid1()).replace('/','_')
os.mkdir(TMPDIR+DIRID)
 
# Printing Global Variables
DEBUG = False
DEBUG_MEM = False
VERBOSE = 1
VERBOSE_TIME = 3
VERBOSE_MEM = 10
OUTPUT_DIGITS = 5
OUTPUT_COLS = 5

# Eigenproblem parameters
DAVIDSON_TOL = 1e-16
DAVIDSON_MAX_ITER = 100
USE_PRECOND = False
ARNOLDI_TOL = 1e-8
ARNOLDI_MAX_ITER = 100

# Memory Global Variables
try:
    import psutil
    _,av,_,_,_,_,_,_,_,_,_ = psutil.virtual_memory()
    MAX_MEMORY = av
except:
    pass
