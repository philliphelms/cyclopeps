"""
Tools for the operators used with 
a projected entangled pair states

Author: Phillip Helms <phelms@caltech.edu>
Date: July 2019

"""

from cyclopeps.tools.utils import *
from cyclopeps.tools.mps_tools import MPS
from scipy.linalg import expm as sla_expm
import copy

def quick_op(op1,op2):
    """
    Convert two local operators (2 legs each)
    into an operator between sites (4 legs)
    """
    return einsum('io,IO->iIoO',op1,op2)

def expm(m,a=1.):
    """
    Take the exponential of a matrix
    """
    m = to_nparray(m)
    m *= a
    m = sla_expm(m)
    m = from_nparray(m)
    return m

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
    # Make a copy if wanted
    exp_ops = ops

    # Exponentiate all operators
    for i in range(len(exp_ops)):
        for j in range(len(exp_ops[i])):
            for k in range(len(exp_ops[j])):
                exp_ops[i][j][k] = expm(exp_ops[i][j][k],a=a)
    
    return exp_ops    

