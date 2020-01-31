"""
Operators for the quantum Heisenberg Model
"""
from cyclopeps.tools.utils import *
from cyclopeps.tools.ops_tools import *
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
                col_ops.append(op())
            elif y == Ny-2:
                col_ops.append(op())
            else:
                col_ops.append(op())
        columns.append(col_ops)

    # Operators within Rows
    rows = []
    for y in range(Ny):
        row_ops = []
        for x in range(Nx-1):
            if x == 0:
                row_ops.append(op())
            elif x == Nx-2:
                row_ops.append(op())
            else:
                row_ops.append(op())
        rows.append(row_ops)

    return [columns,rows]

def op():
    """
    Operator for sites in center of lattice
    """
    op  = quick_op(Sz,Sz)
    op += 1./2 * quick_op(Sp,Sm)
    op += 1./2 * quick_op(Sm,Sp)
    return op
