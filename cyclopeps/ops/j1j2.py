"""
Operators for the quantum J1J2 Heisenberg Model

Currently, operators with next nearest 
neighbors are stored with four MPOs for every
2x2 partitioning of the lattice:

     __             __
    |  o-------------o|
    |o_|           |_||
     |               |
     |               |
     |               |
     |               |
     |_             _|
    || |           | o|
    |o--------------o_|

The first acts on the left and bottom bonds
and contains the (0,0)-(0,1) and (0,0)-(1,0)
interactions.

The second acts on the right and bottom bonds
and contains the (0,0)-(1,1) interactions on 
all units and the (0,1)-(1,1) interaction on
units on the right hand side. (Note that these
are applied to tensors of a flipped PEPS). 

The third acts on the top and right bonds
and always contains the (0,1)-(1,0) interaction
and when on the top boundary it also 
includes the (0,1)-(1,1) interaction.

For the first mpo, the order is:
     __             __
    |1 |           |  |
    |o_|           |__|
     |               
     |               
     |               
     |               
     |_             __
    ||2|           |3 |
    |o--------------o_|

For the second MPO, the order is:
     __             __
    |  |           | 1|
    |__|           |o_|
                    |
                    |
                    |
                    |
     __             |_
    | 3|           ||2|
    |_o-------------o_|

For the third MPO, the order is:
     __             __
    |  o-------------o|
    |1_|           |2||
                     |
                     |
                     |
                     |
     __             _|
    |  |           | o|
    |__|           |3_|

Any single site field is included at the (0,0) 
site of the unit cell and is applied only with
the first MPO. At the top boundary, the field
is also applied to the (0,1) site of first MPO
and at the right boundary, the filed is applied
to the (1,0) site of the first MPO. At the top right
unit cell, the field is also included on the (1,1)
site in the third MPO.
"""
from cyclopeps.tools.utils import *
from cyclopeps.tools.ops_tools import *
from cyclopeps.ops.ops import OPS
from cyclopeps.tools.gen_ten import einsum,zeros
from symtensor.settings import load_lib

def return_op(Nx,Ny,j1=1.,j2=0.,bx=0.,bz=0.,sym=None,backend='numpy'):
    """
    Return the operators

    Args:
        Nx : int
            Lattice size in the x direction
        Ny : int
            Lattixe size in the y direction

    Kwargs:
        j1 : float
            The nearest neighbor interaction strength
        j2 : float
            The next nearest neighbor interaction strength
        bx : float
            The strength of the on site x-direction field.
        bz : float
            The strength of the on site z-direction field.

    Returns:
        op
    """
    # Haven't figured out symmetric case for NN interactions
    if sym is not None:
        raise NotImplementedError()

    # Collect relevant operators
    ops = OPS(sym=sym,backend=backend)

    # Get the tensor backend
    if isinstance(backend,str):
        backend = load_lib(backend)
    else:
        backend = backend

    # Loop over columns
    all_cols = []
    for x in range(Nx-1):
        single_col = []
        for y in range(Ny-1):
            # Create a list to hold all the MPOs acting on the 2x2 unit cell
            cell_mpos = []
            
            # Create first MPO
            # The first acts on the left and bottom bonds
            # and contains the (0,0)-(0,1) and (0,0)-(1,0)
            # interactions. The (0,0) site always contains
            # a single site field. At the top boundary, the 
            # field is also applied to the (0,1) site and at
            # the right boundary, the field is also applied
            # to the (1,0) site. 
            #     __             __
            #    |1 |           |  |
            #    |o_|           |__|
            #     |               
            #     |               
            #     |               
            #     |               
            #     |_             __
            #    ||2|           |3 |
            #    |o--------------o_|
            cell_mpos.append(mpo(ops,j1,j2,bx,bz,backend,
                                 interaction01=True,
                                 interaction12=True,
                                 left_field= (y==Ny-2),
                                 center_field=True,
                                 right_field= (x == Nx-2)))
            # Create second MPO
            # The second acts on the right and bottom bonds
            # and contains the (0,0)-(1,1) interactions on 
            # all units and the (0,1)-(1,1) interaction on
            # units on the right hand side. (Note that these
            # are applied to tensors of a flipped PEPS). 
            #     __             __
            #    |  |           | 1|
            #    |__|           |o_|
            #                    |
            #                    |
            #                    |
            #                    |
            #     __             |_
            #    | 3|           ||2|
            #    |_o-------------o_|
            cell_mpos.append(mpo(ops,j1,j2,bx,bz,backend,
                                 interaction02=True))
            # Create third MPO
            # The third acts on the top and right bonds
            # and always contains the (0,1)-(1,0) interaction
            # and when on the top boundary it also 
            # includes the (0,1)-(1,1) interaction.
            # A field is added to the (1,1) site
            # when at the rightmost and
            # uppermost unit cell.
            #     __             __
            #    |  o-------------o|
            #    |1_|           |2||
            #                     |
            #                     |
            #                     |
            #                     |
            #     __             _|
            #    |  |           | o|
            #    |__|           |3_|
            cell_mpos.append(mpo(ops,j1,j2,bx,bz,backend,
                                 interaction02=True,
                                 interaction01=(y==Ny-2),
                                 interaction12=(x==Nx-2),
                                 center_field = ((y==Ny-2) and (x==Nx-2))))
            
            # Add mpos on the 2x2 square into the column's interactions
            single_col.append(cell_mpos)

        # Add columns interaction to list of all interactions
        all_cols.append(single_col)

    return all_cols

def mpo(ops,j1,j2,bx,bz,backend,
        left_field=False,center_field=False,right_field=False,
        interaction01=False,interaction12=False,interaction02=False):
    """
    MPO creation for interaction on left and bottom 
    edges of the 2x2 square
    """
    # Collect operators
    Sx = ops.Sx.ten
    Sz = ops.Sz.ten
    Sp = ops.Sp.ten
    Sm = ops.Sm.ten
    I  = ops.I.ten
    # First (top left) site -------------------------
    op1 = zeros((2,2,5),sym=None,backend=backend)
    op1[:,:,1] = j1*Sz
    op1[:,:,2] = j1/2.*Sp
    op1[:,:,3] = j1/2.*Sm
    op1[:,:,4] = I
    # Add field:
    if left_field:
        op1[:,:,0] = bx*Sx + bz*Sz
    # Central (bottom left) site --------------------
    op2 = zeros((5,2,2,5),sym=None,backend=backend)
    # Left column
    op2[0,:,:,0] = I
    if interaction01:
        op2[1,:,:,0] = Sz
        op2[2,:,:,0] = Sm
        op2[3,:,:,0] = Sp
    # Fill center
    if interaction02:
        op2[1,:,:,1] = j2/j1*I
        op2[2,:,:,2] = j2/j1*I
        op2[3,:,:,3] = j2/j1*I
    # Bottom row
    if interaction12:
        op2[4,:,:,1] = j1*Sz
        op2[4,:,:,2] = j1/2.*Sp
        op2[4,:,:,3] = j1/2.*Sm
    op2[4,:,:,4] = I
    # Add field:
    if center_field:
        op2[4,:,:,0] = bx*Sx + bz*Sz
    # Third (bottom right) site -------------------
    op3 = zeros((5,2,2),sym=None,backend=backend)
    op3[0,:,:] = I
    op3[1,:,:] = Sz
    op3[2,:,:] = Sm
    op3[3,:,:] = Sp
    # Add field:
    if right_field:
        op3[4,:,:] = bx*Sx + bz*Sz
    # Put into a list -----------------------------
    mpo = [op1,op2,op3]
    return mpo
