"""
Suzuki-Trotter decomposition of the tilted
generator for the 2D simple exclusion process.
"""

from cyclopeps.tools.utils import *
from cyclopeps.ops.ops import *
from cyclopeps.tools.ops_tools import *
from numpy import float_
from numpy import exp
import collections

def return_dens_op(Nx,Ny,top=True):
    """
    Return the operators to give local densities

    Args:
        Nx : int
            Lattice size in the x direction
        Ny : int
            Lattixe size in the y direction

    Kwargs:
        top : bool
            Whether to measure density on top or bottom site

    Returns:
        ops
    """

    # Operators within columns
    columns = []
    for x in range(Nx):
        # Create operator for single column
        col_ops = []
        for y in range(Ny-1):
            # Create operator for intereaction between sites (y,y+1)
            if top:
                op = quick_op(I,n)
            else:
                op = quick_op(n,I)
            # Add to list of column operators
            col_ops.append(op)
        # Add column of operators to list of columns
        columns.append(col_ops)

    # Operators within Rows
    rows = []
    for y in range(Ny):
        # Create operator for single row
        row_ops = []
        for x in range(Nx-1):
            # Create operator for interaction between sites (x,x+1)
            if top:
                op = quick_op(I,n)
            else:
                op = quick_op(n,I)
            # Add to list of row operators
            row_ops.append(op)
        # Add row of operators to list of rows
        rows.append(row_ops)

    # Return Results
    return [columns,rows]
