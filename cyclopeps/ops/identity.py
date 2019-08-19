"""
Operators for the Ising Transverse Field Model
"""
from cyclopeps.tools.utils import *
from cyclopeps.ops.ops import *

def return_op(Nx,Ny):
    """
    Return the operators

    Args:
        Nx : int
            Lattice size in the x direction
        Ny : int
            Lattixe size in the y direction

    Returns:
        op
    """
    # Operators within columns
    columns = []
    for x in range(Nx):
        col_ops = []
        for y in range(Ny-1):
            if y == 0:
                col_ops.append(op0())
            elif y == Ny-2:
                col_ops.append(opN())
            else:
                col_ops.append(op())
        columns.append(col_ops)

    # Operators within Rows
    rows = []
    for y in range(Ny):
        row_ops = []
        for x in range(Nx-1):
            if x == 0:
                row_ops.append(op0())
            elif x == Nx-2:
                row_ops.append(opN())
            else:
                row_ops.append(op())
        rows.append(row_ops)

    return [columns,rows]

def op():
    """
    Operator for sites in center of lattice
    """
    op = 0.5*einsum('io,IO->iIoO',I,I)
    return op

def op0():
    """
    Operator for sites (0,1)
    """
    op = 0.25*einsum('io,IO->iIoO',I,I)
    return op

def opN(params):
    """
    Operator for site (N-1,N)
    """
    op = 0.5*einsum('io,IO->iIoO',I,I)
    return op
