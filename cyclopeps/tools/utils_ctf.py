"""
Linear Algebra Tools

Author: Phillip Helms <phelms@caltech.edu>
Date: June 2019

"""
from cyclopeps.tools.params import *# import params
import ctf 
from numpy import array as nparray
from numpy import expand_dims as npexpand_dims
from numpy import prod as npprod
from numpy import sqrt as npsqrt
from numpy import log2 as nplog2
from numpy.linalg import inv as npinv
from numpy import complex128
from numpy import complex64
from numpy import complex_
from numpy import float_
from numpy import real as npreal
from numpy import imag as npimag

# Tensor Allocation
def array(tens,dtype=None,copy=True,subok=False,ndimin=0):
    ten = nparray(tens)
    return ctf.from_nparray(ten)
eye        = ctf.eye
ones       = ctf.ones
rand       = ctf.random.random
def rand(shape,p=None,dtype=None):
    if dtype == complex128 or dtype == complex64:
        ten_real1 = ctf.zeros(shape,dtype=float_)
        ten_real1.fill_random()
        ten_real2 = ctf.zeros(shape,dtype=float_)
        ten_real2.fill_random()
        ten = ctf.zeros(shape,dtype=complex_)
        ten += ten_real1
        ten += 1.j*ten_real2
    else:
        ten = ctf.zeros(shape,dtype=dtype,sp=False)
        ten.fill_random()
    return ten
zeros      = ctf.zeros
zeros_like = ctf.zeros_like

# Linear Algebra
abss       = ctf.abs
dot        = ctf.dot
einsum     = ctf.einsum
qr         = ctf.qr
summ       = ctf.sum
svd        = ctf.svd
eigh       = ctf.eigh
#def dot(a,b):
#    res = ctf.dot(a,b)
#    if (prod(res.shape) == 1) or (len(res.shape) == 0):
#        return summ(res)
#    else:
#        return res
def log2(a):
    return ctf.from_nparray(nplog2(ctf.to_nparray(a)))
def prod(a):
    return npprod(ctf.to_nparray(a))
def sqrt(a):
    if hasattr(a,'shape'):
        if (prod(a.shape) == 1) or (len(a.shape) == 0):
            return npsqrt(ctf.to_nparray(a))
        else:
            return ctf.from_nparray(npsqrt(ctf.to_nparray(a)))
    else:
        return npsqrt(a)
def inv(ten):
    ten = to_nparray(ten)
    ten = npinv(ten)
    return from_nparray(ten)

# Tensor Manipulation
conj       = ctf.conj
diag       = ctf.diag
diagonal   = ctf.diagonal
ravel      = ctf.ravel

def imag(val):
    try:
        val = ctf.imag(val)
    except:
        val = npimag(val)
    return val

def real(val):
    try:
        val = ctf.real(val)
    except:
        val = npreal(val)
    return val

reshape    = ctf.reshape
transpose  = ctf.transpose
to_nparray = ctf.to_nparray
#from_nparray=ctf.from_nparray
from_nparray=ctf.astensor
take       = ctf.take
import inspect
print(inspect.getmodule(take).__file__)
def expand_dims(ten,ax):
    ten = ctf.to_nparray(ten)
    ten = npexpand_dims(ten,ax)
    return from_nparray(ten)
# Save & Load Tensors
def save_ten(ten,fname):
    ten.write_to_file(fname)
def load_ten(dim,fname,dtype=None,):
    ten = ctf.zeros(dim,dtype=dtype)
    ten.read_from_file(fname)
    return ten
def argsort(seq):
    return sorted(range(seq.shape[0]), key=seq.__getitem__)
# Overwrite functions that are different for sparse tensors
if USE_SPARSE:
    diag = ctf.spdiag
    eye = ctf.speye
    def rand(shape,p=None,dtype=None):
        if dtype == complex128 or dtype == complex64:
            ten_real = ctf.zeros(shape,dtype=float_,sp=True)
            ten_real.fill_sp_random()
            ten = ctf.zeros(shape,dtype=dtype,sp=True)
            ten = ten_real+0.j
        else:
            ten = ctf.zeros(shape,dtype=dtype,sp=True)
            ten_real.fill_sp_random()
        return ten
    def zeros(shape,dtype=None):
        return ctf.zeros(shape,dtype=dtype,sp=True)
    def load_ten(dim,fname,dtype=None,sp=True):
        ten = zeros(dim,dtype=dtype,sp=sp)
        ten.read_from_file(fname)
        return ten
