#from pyscf.lib import logger
from mpi4py import MPI
import ctf
import numpy as np
import scipy.linalg as sla

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#class Logger(logger.Logger):
#    def __init__(self, stdout, verbose):
#        if rank == 0:
#            logger.Logger.__init__(self, stdout, verbose)
#        else:
#            logger.Logger.__init__(self, stdout, 0)

def static_partition(tasks):
    segsize = (len(tasks)+size-1) // size
    start = rank * segsize
    stop = min(len(tasks), start+segsize)
    return tasks[start:stop]

NAME='ctf'
einsum = ctf.einsum
astensor = ctf.astensor
zeros = ctf.zeros
empty = ctf.empty
ones = ctf.ones
eye = ctf.eye
rint = ctf.rint
random = ctf.random.random
reshape = ctf.reshape
transpose = ctf.transpose
array = ctf.array
hstack = ctf.hstack
vstack = ctf.vstack
dot = ctf.dot
#svd = ctf.svd
#qr = ctf.qr
diag = ctf.diag
abs = ctf.abs
sum = ctf.sum
eye = ctf.eye


def non_zeros(a):
    return a.read_all_nnz()[0]

def copy(a):
    return a.copy()

def write_all(a, ind, fill):
    a.write(ind, fill)
    return a

def write_single(a, ind, fill):
    if rank==0:
        a.write(ind, fill)
    else:
        a.write([],[])
    return a
write = write_all
def to_nparray(a):
    return a.to_nparray()

def norm(a):
    return a.norm2()


def find_less(a, threshold):
    c = a.sparsify(threshold)
    idx, vals = a.read_all_nnz()
    return idx

# Problem Functions
def append(a,b):
    return ctf.from_nparray(np.append(ctf.to_nparray(a),ctf.to_nparray(b)))
def qr(a):
    a = ctf.to_nparray(a)
    q,r = np.linalg.qr(a)
    q = ctf.from_nparray(q)
    r = ctf.from_nparray(r)
    return q,r
def svd(a,full_matrices=False):
    a = ctf.to_nparray(a)
    u,s,v = np.linalg.svd(a,full_matrices=full_matrices)
    u = ctf.from_nparray(u)
    s = ctf.from_nparray(s)
    v = ctf.from_nparray(v)
    return u,s,v
def eigh(a):
    a = ctf.to_nparray(a)
    u,v = np.linalg.eigh(a)
    u = ctf.from_nparray(u)
    v = ctf.from_nparray(v)
    return u,v
def sqrt(a):
    return ctf.from_nparray(np.sqrt(ctf.to_nparray(a)))
def log2(a):
    return ctf.from_nparray(np.log2(ctf.to_nparray(a)))
def isnan(a):
    return False
    #return ctf.from_nparray(np.isnan(ctf.to_nparray(a)))
def expm(a):
    return ctf.from_nparray(sla.expm(ctf.to_nparray(a)))
def inv(a):
    return ctf.from_nparray(np.linalg.pinv(ctf.to_nparray(a)))
def max(a):
    return ctf.from_nparray(np.max(ctf.to_nparray(a)))
def min(a):
    return ctf.from_nparray(np.min(ctf.to_nparray(a)))

#isnan = ctf.isnan
