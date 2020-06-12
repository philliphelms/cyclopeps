import numpy as np
import scipy.linalg as sla
#from pyscf.lib.logger import Logger
BACKEND = 'numpy'

einsum = np.einsum
astensor = np.asarray
zeros = np.zeros
empty = np.empty
array = np.array
ones = np.ones
rint = np.rint
random = np.random.random
norm = np.linalg.norm
svd = np.linalg.svd
qr = np.linalg.qr
eigh = np.linalg.eigh
def inv(mat):
    try:
        return np.linalg.inv(mat)
    except:
        return np.linalg.pinv(mat)
#inv = np.linalg.pinv
def pinv(mat):
    return np.linalg.pinv(mat)
dot = np.dot
diag = np.diag
reshape = np.reshape
transpose = np.transpose
abs = np.abs
sum = np.sum
eye = np.eye
max = np.max
min = np.min
append = np.append
sqrt = np.sqrt
log2 = np.log2
eye = np.eye
hstack = np.hstack
vstack = np.vstack
isnan = np.isnan
expm = sla.expm

def non_zeros(a):
    idx = np.where(a.ravel()!=0)
    return idx[0]
def copy(a):
    return a.copy()

def write_all(a, ind, fill):
    a.put(ind, fill)
    return a

write = write_single = write_all

def to_nparray(a):
    return a

def find_less(a, threshold):
    idx = np.where(a.ravel()<threshold)[0]
    return idx
