"""
Tools for the operators used with 
a projected entangled pair states

Author: Phillip Helms <phelms@caltech.edu>
Date: July 2019

"""

from cyclopeps.tools.utils import *
from cyclopeps.tools.mps_tools import MPS
from cyclopeps.tools.gen_ten import einsum
import copy

def quick_op(op1,op2):
    """
    Convert two local operators (2 legs each)
    into an operator between sites (4 legs)
    """
    #return einsum('io,IO->iIoO',op1,op2)
    return einsum('io,IO->oOiI',op1,op2)

def exp_gate_nosym(h,a=1.):
    """
    Take the exponential of a local two site
    gate for a non-symmetric tensor
    """
    # Get correct library
    lib = h.backend
    # Create tensors to save results
    eh = h.copy()
    res = h.copy().ten
    # Reshape into a matrix
    d = res.shape[0]
    res = lib.reshape(res,(d**2,d**2))
    # Take exponential of matrix
    res *= a
    res = lib.expm(res)
    # Reshape back into a two-site gate
    res = lib.reshape(res,(d,d,d,d))
    # Put result into a gen_ten
    eh.ten = res
    return eh

def exp_gate_sym(h,a=1.):
    """
    Take the exponential of a local two site gate
    for a gate that is a symtensor
    """
    # Get correct library
    lib = h.backend
    # Create tensors to save results
    eh = h.copy()
    res = h.copy().ten.make_sparse()
    d = res.shape[0]
    # Reshape into a matrix for exponentiation
    res = lib.reshape(res,(d**2,d**2))
    # Take exponential of matrix
    res *= a
    res = lib.expm(res)
    # Reshape back into a two-site gate
    res = lib.reshape(res,(d,d,d,d,1,1,1,1))
    # Put back into a symtensor
    delta = h.ten.get_irrep_map()
    res = lib.einsum('ABCDabcd,ABCD->ABCabcd',res,delta)
    eh.ten.array = res
    # Check that we haven't messed up the symmetry
    eh.ten.enforce_sym()
    return eh

def exp_gate(h,a=1.):
    """
    Take the exponential of a local two site gate
    """
    if h.sym is None:
        return exp_gate_nosym(h,a=a)
    else:
        return exp_gate_sym(h,a=a)

def take_exp(ops,a=1.):
    """
    Take the exponential of all operators
    in the ops lists

    Args:
        ops : list of list of lists
            The trotter decomposed operators 

    Kwargs:
        a : float
            A constant, taking exp(a*M)

    Returns:
        exp_ops : lost of list of lists
            The resulting exponentiated operators
    """
    # Get correct library
    lib = ops[0][0][0].backend
    # Make a copy if wanted
    exp_ops = ops

    # Exponentiate all operators
    for i in range(len(exp_ops)):
        for j in range(len(exp_ops[i])):
            for k in range(len(exp_ops[j])):
                exp_ops[i][j][k] = lib.expm(exp_ops[i][j][k],a=a)
    
    return exp_ops    

def ops_conj_trans(ops,copy=True):
    """
    Return the conjugate transpose of the input ops

    Args:
        ops : Array of local gate operators
            An array that contains a list of local gate operators, 

    Kwargs:
        copy : bool
            Whether to copy the ops before conjugating

    Returns:
        mpoList : 1D Array
            A conjugated, transposed version of the input mpo
    """
    mpiprint(5,'Taking the conjugate transpose of the ops')

    # Copy the ops (if desired)
    if copy: ops = copy_ops(ops)

    # Sweep through all operators, conjugating and transposing each
    for i in range(len(ops)):
        for j in range(len(ops[i])):
            for k in range(len(ops[i][j])):
                ops[i][j][k] = ops[i][j][k].transpose([2,3,0,1]).conj()
    
    # return result
    return ops

def copy_ops(ops):
    """
    Copy a set of local gate operators

    Args:
        ops : list of list of list
            The trotter decomposed operators

    Returns:
        ops : list of list of list
            A copy of the input operators
    """
    mpiprint(7,'Copying ops')

    # Create list to copy ops into
    opsCopy = []

    # Loop over vertical operators
    vert_ops = []
    for i in range(len(ops[0])):
        tmp = [None]*len(ops[0][i])
        for j in range(len(ops[0][i])):
            if ops[0][i][j] is not None:
                tmp[j] = ops[0][i][j].copy()
        vert_ops.append(tmp)
    opsCopy.append(vert_ops)

    # Loop over horizontal operators
    horz_ops = []
    for i in range(len(ops[1])):
        tmp = [None]*len(ops[1][i])
        for j in range(len(ops[1][i])):
            if ops[1][i][j] is not None:
                tmp[j] = ops[1][i][j].copy()
        horz_ops.append(tmp)
    opsCopy.append(horz_ops)
    
    return opsCopy
