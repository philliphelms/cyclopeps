"""
Operators for the Ising Transverse Field Model
"""
from cyclopeps.tools.utils import *
from cyclopeps.tools.ops_tools import *
from cyclopeps.ops.ops import *

def return_op(Nx,Ny,params):
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
                params[0] = spin-spin interaction
                params[1] = Magnetic Field

    Returns:
        op
    """
    # Operators within columns
    columns = []
    for x in range(Nx):
        col_ops = []
        for y in range(Ny-1):
            if y == 0:
                col_ops.append(op0(params))
            elif y == Ny-2:
                col_ops.append(opN(params))
            else:
                col_ops.append(op(params))
        columns.append(col_ops)

    # Operators within Rows
    rows = []
    for y in range(Ny):
        row_ops = []
        for x in range(Nx-1):
            if x == 0:
                row_ops.append(op0(params))
            elif x == Nx-2:
                row_ops.append(opN(params))
            else:
                row_ops.append(op(params))
        rows.append(row_ops)

    return [columns,rows]

def op(params):
    """
    Operator for sites in center of lattice
    """
    op  = 1./4. * params[0]*quick_op(Z,Z)
    op += 1./4. * params[1]*quick_op(X,X)
    op -= 1./4. * params[1]*quick_op(Y,Y)
    return op

def op0(params):
    """
    Operator for sites (0,1)
    """
    op  = 1./4. * params[0]*quick_op(Z,Z)
    op += 1./4. * params[1]*quick_op(X,X)
    op -= 1./4. * params[1]*quick_op(Y,Y)
    return op

def opN(params):
    """
    Operator for site (N-1,N)
    """
    op  = 1./4. * params[0]*quick_op(Z,Z)
    op += 1./4. * params[1]*quick_op(X,X)
    op -= 1./4. * params[1]*quick_op(Y,Y)
    return op
