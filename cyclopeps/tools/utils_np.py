"""
Linear Algebra Tools

Author: Phillip Helms <phelms@caltech.edu>
Date: June 2019

"""
from cyclopeps.tools.params import *# import params
import numpy as np


# -------------------------------------------------
# Tensor Allocation
array      = np.array
eye        = np.eye
ones       = np.ones
rand       = np.random.random
zeros      = np.zeros
zeros_like = np.zeros_like

def rand(dim,dtype=None):
    return np.random.random(dim).astype(dtype)
#    if len(dim) == 1:
#        return np.random.random(dim[0]).astype(dtype)#+1.j*np.random.random(dim).astype(dtype)
#    if len(dim) == 2:
#        return np.random.random(dim[0],dim[1]).astype(dtype)#+1.j*np.random.random(dim).astype(dtype)
#    if len(dim) == 3:
#        return np.random.random(dim[0],dim[1],dim[2]).astype(dtype)#+1.j*np.random.random(dim).astype(dtype)
#    if len(dim) == 4:
#        return np.random.random(dim[0],dim[1],dim[2],dim[3]).astype(dtype)#+1.j*np.random.random(dim).astype(dtype)
#    if len(dim) == 5:
#        return np.random.random(dim[0],dim[1],dim[2],dim[3],dim[4]).astype(dtype)#+1.j*np.random.random(dim).astype(dtype)
#    if len(dim) == 6:
#        return np.random.random(dim[0],dim[1],dim[2],dim[3],dim[4],dim[5]).astype(dtype)#+1.j*np.random.random(dim).astype(dtype)
#    if len(dim) == 7:
#        return np.random.random(dim[0],dim[1],dim[2],dim[3],dim[4],dim[5],dim[6]).astype(dtype)#+1.j*np.random.random(dim).astype(dtype)

def save_ten(ten,fname):
    np.save(fname,ten)

def load_ten(dim,fname,dtype=None):    
    return np.load(fname+'.npy')

def from_nparray(arr):
    return arr

def to_nparray(arr):
    return arr

# -------------------------------------------------
# Linear Algebra
abss       = np.abs
dot        = np.dot
einsum     = np.einsum
qr         = np.linalg.qr
summ       = np.sum
prod       = np.prod
sqrt       = np.sqrt
log2       = np.log2
eigh       = np.linalg.eigh
def svd(ten):
    return np.linalg.svd(ten,full_matrices=False)

# -------------------------------------------------
# Tensor Manipulation
conj       = np.conj
diag       = np.diag
diagonal   = np.diagonal
imag       = np.imag
ravel      = np.ravel
real       = np.real
reshape    = np.reshape
transpose  = np.transpose
expand_dims= np.expand_dims
take       = np.take
argsort    = np.argsort
