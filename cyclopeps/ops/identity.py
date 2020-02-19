"""
Operators for the Ising Transverse Field Model
"""
from cyclopeps.tools.utils import *
from cyclopeps.ops.ops import OPS
from cyclopeps.tools.gen_ten import einsum

def return_op(Nx,Ny,sym=None,backend='numpy'):
    """
    Return the operators

    Args:
        Nx : int
            Lattice size in the x direction
        Ny : int
            Lattixe size in the y direction

    Kwargs:
        sym : str
            The desired tensor symmetry,
            default is None
        backend : str
            The desired tensor backend for the
            operators

    Returns:
        op
    """
    # Collect relevant operators
    ops = OPS(sym=sym,backend=backend)
    # Operators within columns
    columns = []
    for x in range(Nx):
        col_ops = []
        for y in range(Ny-1):
            if y == 0:
                col_ops.append(op0(ops))
            elif y == Ny-2:
                col_ops.append(opN(ops))
            else:
                col_ops.append(op(ops))
        columns.append(col_ops)

    # Operators within Rows
    rows = []
    for y in range(Ny):
        row_ops = []
        for x in range(Nx-1):
            if x == 0:
                row_ops.append(op0(ops))
            elif x == Nx-2:
                row_ops.append(opN(ops))
            else:
                row_ops.append(op(ops))
        rows.append(row_ops)

    return [columns,rows]

def op(ops):
    """
    Operator for sites in center of lattice

    Args:
        ops : OPS object
            An object holding required operators
    """
    I = ops.I
    op = 0.5*einsum('io,IO->iIoO',I,I)
    return op

def op0(ops):
    """
    Operator for sites (0,1)
    
    Args:
        ops : OPS object
            An object holding required operators
    """
    I = ops.I
    op = 0.25*einsum('io,IO->iIoO',I,I)
    return op

def opN(ops):
    """
    Operator for site (N-1,N)
    
    Args:
        ops : OPS object
            An object holding required operators
    """
    I = ops.I
    op = 0.5*einsum('io,IO->iIoO',I,I)
    return op
