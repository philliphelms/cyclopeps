"""
Operators for the Ising Transverse Field Model
"""
from cyclopeps.tools.utils import *
from cyclopeps.tools.ops_tools import *
from cyclopeps.ops.ops import OPS
from cyclopeps.tools.gen_ten import einsum

def return_op(Nx,Ny,params,sym=None,backend='numpy'):
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
            if y == 0:
                col_ops.append(op0(params,ops))
            elif y == Ny-2:
                col_ops.append(opN(params,ops))
            else:
                col_ops.append(op(params,ops))
        columns.append(col_ops)

    # Operators within Rows
    rows = []
    for y in range(Ny):
        row_ops = []
        for x in range(Nx-1):
            if x == 0:
                row_ops.append(op0(params,ops))
            elif x == Nx-2:
                row_ops.append(opN(params,ops))
            else:
                row_ops.append(op(params,ops))
        rows.append(row_ops)

    return [columns,rows]

def op(params,ops):
    """
    Operator for sites in center of lattice
    """
    # Collect needed operators
    Z = ops.Z
    X = ops.X
    Y = ops.Y
    # Create operator
    op  = 1./4. * params[0]*quick_op(Z,Z)
    op += 1./4. * params[1]*quick_op(X,X)
    op -= 1./4. * params[1]*quick_op(Y,Y)
    return op

def op0(params,ops):
    """
    Operator for sites (0,1)
    """
    # Collect needed operators
    Z = ops.Z
    X = ops.X
    Y = ops.Y
    # Create operator
    op  = 1./4. * params[0]*quick_op(Z,Z)
    op += 1./4. * params[1]*quick_op(X,X)
    op -= 1./4. * params[1]*quick_op(Y,Y)
    return op

def opN(params,ops):
    """
    Operator for site (N-1,N)
    """
    # Collect needed operators
    Z = ops.Z
    X = ops.X
    Y = ops.Y
    # Create operator
    op  = 1./4. * params[0]*quick_op(Z,Z)
    op += 1./4. * params[1]*quick_op(X,X)
    op -= 1./4. * params[1]*quick_op(Y,Y)
    return op
