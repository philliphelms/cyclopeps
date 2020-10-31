#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#
"""Numpy backend for Symtensor"""
from mkl_interface import einsum_batched_matmul
import numpy as np
BACKEND = 'blas'

einsum = np.einsum
def einsum(sub, *operands, **kwargs):
    try:
        out = einsum_batched_matmul(sub, *operands)
    except Exception as e:
        out = np.einsum(sub, *operands, **kwargs)
    return out
#einsum = einsum_batched_matmul
astensor = np.asarray
zeros = np.zeros
empty = np.empty
ones = np.ones
rint = np.rint
random = np.random.random
norm = np.linalg.norm
qr = np.linalg.qr
dot = np.dot
diag = np.diag
reshape = np.reshape
eye = np.eye
hstack = np.hstack
vstack = np.vstack
append = np.append
sqrt = np.sqrt
log2 = np.log2
expm = sla.expm
save = np.save
load = np.load

def non_zeros(a):
    idx = np.where(a.ravel()!=0)
    return idx[0]
def copy(a):
    return a.copy()
def pinv(mat):
    return np.linalg.pinv(mat)

def write_all(a, ind, fill):
    a.put(ind, fill)
    return a

write = write_single = write_all

def to_nparray(a):
    return a

def find_less(a, threshold):
    idx = np.where(a.ravel()<threshold)[0]
    return idx
def svd(mat,full_matrices=False):
    try:
        return np.linalg.svd(mat,full_matrices=False)
    except:
        return sla.svd(mat,lapack_driver='gesvd')
