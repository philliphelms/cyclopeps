"""
Operators for the Frederick-Anderson One Spin 
Facilitated Kinetically Constrained Model
"""
from cyclopeps.tools.utils import *
from cyclopeps.ops.ops import *
from numpy import exp
from cyclopeps.tools.ops_tools import *

def return_op(Nx,Ny,params,hermitian=False):
    """
    Return the operators

    Args:
        Nx : int
            Lattice size in the x direction
        Ny : int
            Lattixe size in the y direction
        params : 1D Array
            The parameters for the hamiltonian.
            Here,
                params[0] = c      : Spin flip rate, c = beta = 1/T = ln(1-c)/c
                params[1] = \lambda: Bias towards higher activity 

    Returns:
        op
    """
    # Operators within columns
    columns = []
    for x in range(Nx):
        col_ops = []
        for y in range(Ny-1):
            if (x == 0) and (y == Ny-2):
                col_ops.append(corner_op(params,hermitian=hermitian))
            else:
                col_ops.append(op(params,hermitian=hermitian))
        columns.append(col_ops)

    # Operators within Rows
    rows = []
    for y in range(Ny):
        row_ops = []
        for x in range(Nx-1):
            if (x == 0) and (y == Ny-1):
                row_ops.append(corner_op(params,hermitian=hermitian))
            else:
                row_ops.append(op(params,hermitian=hermitian))
        rows.append(row_ops)

    return [columns,rows]

def op(params,hermitian=False):
    """
    Operator for sites in center of lattice
    
    Interaction given by equation:
    W = n_{i-1} * (  c e^{-\lambda} \sigma_i^{+}
                   + (1-c) e^{-\lambda} \sigma_i^{-}
                   - c (1-n_i)
                   - (1-c) n_i )
        n_{i}   * (  c e^{-\lambda} \sigma_{i-1}^{+}
                   + (1-c) e^{-\lambda} \sigma_{i-1}^{-}
                   - c (1-n_{i-1})
                   - (1-c) n_{i-1} )
    """
    c = params[0]
    s = params[1]
    if hermitian:
        # Add 'east' interaction
        op  = exp(-s)*sqrt(c*(1.-c)) * quick_op(n,X)
        op -= c * quick_op(n,v)
        op -= (1.-c) * quick_op(n,n)
        op = -op
    else:
        # Add 'east' interaction
        op  = c*exp(-s) * quick_op(n,Sp)
        op += (1.-c)*exp(-s) * quick_op(n,Sm)
        op -= c * quick_op(n,v)
        op -= (1.-c) * quick_op(n,n)
        op = -op
    # Return result
    return op

def corner_op(params,hermitian=False):
    """
    Operator to keep top left occpied
    """
    c = params[0]
    s = params[1]
    if hermitian:
        # Add 'east' interaction
        op  = exp(-s)*sqrt(c*(1.-c)) * quick_op(I,X)
        op -= c * quick_op(I,v)
        op -= (1.-c) * quick_op(I,n)
        op = -op
    else:
        # Add 'east' interaction
        op  = c*exp(-s) * quick_op(I,Sp)
        op += (1.-c)*exp(-s) * quick_op(I,Sm)
        op -= c * quick_op(I,v)
        op -= (1.-c) * quick_op(I,n)
        op = -op
    # Return result
    return op

def return_act_op(Nx,Ny,params,hermitian=False):
    """
    Operators to compute the activity

    Args:
        Nx : int
            Lattice size in the x direction
        Ny : int
            Lattixe size in the y direction
        params : 1D Array
            The parameters for the hamiltonian.
            Here,
                params[0] = c      : Spin flip rate, c = beta = 1/T = ln(1-c)/c
                params[1] = \lambda: Bias towards higher activity 

    Returns:
        op
    """
    # Operators within columns
    columns = []
    for x in range(Nx):
        col_ops = []
        for y in range(Ny-1):
            if (x == 0) and (y == Ny-2):
                col_ops.append(corner_act_op(params,hermitian=hermitian))
            else:
                col_ops.append(act_op(params,hermitian=hermitian))
        columns.append(col_ops)

    # Operators within Rows
    rows = []
    for y in range(Ny):
        row_ops = []
        for x in range(Nx-1):
            if (x == 0) and (y == Ny-1):
                row_ops.append(corner_act_op(params,hermitian=hermitian))
            else:
                row_ops.append(act_op(params,hermitian=hermitian))
        rows.append(row_ops)

    return [columns,rows]

def act_op(params,hermitian=False):
    """
    Operator for sites in center of lattice
    
    Interaction given by equation:
    W = n_{i-1} * (  c e^{-\lambda} \sigma_i^{+}
                   + (1-c) e^{-\lambda} \sigma_i^{-}
                   - c (1-n_i)
                   - (1-c) n_i )
        n_{i}   * (  c e^{-\lambda} \sigma_{i-1}^{+}
                   + (1-c) e^{-\lambda} \sigma_{i-1}^{-}
                   - c (1-n_{i-1})
                   - (1-c) n_{i-1} )
    """
    c = params[0]
    s = params[1]
    if hermitian:
        # Add 'east' interaction
        op  = exp(-s)*sqrt(c*(1.-c)) * quick_op(n,X)
        op = -op
    else:
        # Add 'east' interaction
        op  = c*exp(-s) * quick_op(n,Sp)
        op += (1.-c)*exp(-s) * quick_op(n,Sm)
        op = -op
    # Return result
    return op

def corner_act_op(params,hermitian=False):
    """
    Operator to keep top left occpied
    """
    c = params[0]
    s = params[1]
    if hermitian:
        # Add 'east' interaction
        op  = exp(-s)*sqrt(c*(1.-c)) * quick_op(I,X)
        op = -op
    else:
        # Add 'east' interaction
        op  = c*exp(-s) * quick_op(I,Sp)
        op += (1.-c)*exp(-s) * quick_op(I,Sm)
        op = -op
    # Return result
    return op
