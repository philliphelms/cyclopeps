"""
Operators for the Heisenberg Model
"""
from cyclopeps.tools.utils import *
from cyclopeps.tools.ops_tools import *
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
        params : 1D Array
            The parameters for the hamiltonian.
            Here,
                params[0] = spin-spin interaction
                params[1] = Magnetic Field

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
            col_ops.append(op(ops))
        columns.append(col_ops)

    # Operators within Rows
    rows = []
    for y in range(Ny):
        row_ops = []
        for x in range(Nx-1):
            row_ops.append(op(ops))
        rows.append(row_ops)

    return [columns,rows]

def op(ops):
    """
    Operator for sites in center of lattice
    """
    # Collect needed operators
    Sz = ops.Sz
    Sp = ops.Sp
    Sm = ops.Sm
    # Create operator
    op  = quick_op(Sz,Sz)
    op += 1./2. * quick_op(Sp,Sm)
    op += 1./2. * quick_op(Sm,Sp)
    return op
