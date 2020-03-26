"""
Operators for the quantum J1J2 Model

Currently, operators with next nearest 
neighbors are stored with two MPOs for every
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

For bottom left mpo, order is:
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

For the top right mpo, order is:
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

"""
from cyclopeps.tools.utils import *
from cyclopeps.tools.ops_tools import *
from cyclopeps.ops.ops import OPS
from cyclopeps.tools.gen_ten import einsum,zeros
from symtensor.settings import load_lib

def return_op(Nx,Ny,j1=1.,j2=0.,sym=None,backend='numpy'):
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
            # Convention: 
            #   In the bulk, the left-bottom MPO carries the 
            #   horizontal interactions and the right-top 
            #   MPO carries the vertical interactions
            #   * This must be changed at left and top boundaries

            # Create a list to hold all the MPOs acting on the 2x2 square
            square_mpos = []

            # Add left-bottom interaction as an MPO
            if x == 0:
                square_mpos.append(mpo(ops,j1,j2,backend,edge=True))
            else:
                square_mpos.append(mpo(ops,j1,j2,backend))

            # Add right-top interaction as an MPO
            if y == Ny-2:
                square_mpos.append(mpo(ops,j1,j2,backend,edge=True))
            else:
                square_mpos.append(mpo(ops,j1,j2,backend))
            
            # Add mpos on the 2x2 square into the column's interactions
            single_col.append(square_mpos)

        # Add columns interaction to list of all interactions
        all_cols.append(single_col)

    return all_cols

def mpo(ops,j1,j2,backend,edge=False):
    """
    MPO creation for interaction on left and bottom 
    edges of the 2x2 square
    """
    # Collect operators
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
    # Central (bottom left) site --------------------
    op2 = zeros((5,2,2,5),sym=None,backend=backend)
    # Left column
    op2[0,:,:,0] = I
    if edge:
        op2[1,:,:,0] = Sz
        op2[2,:,:,0] = Sm
        op2[3,:,:,0] = Sp
    # Fill center
    op2[1,:,:,1] = j2/j1*I
    op2[2,:,:,2] = j2/j1*I
    op2[3,:,:,3] = j2/j1*I
    # Bottom row
    op2[4,:,:,1] = j1*Sz
    op2[4,:,:,2] = j1/2.*Sp
    op2[4,:,:,3] = j1/2.*Sm
    op2[4,:,:,4] = I
    # Third (bottom right) site -------------------
    op3 = zeros((5,2,2),sym=None,backend=backend)
    op3[0,:,:] = I
    op3[1,:,:] = Sz
    op3[2,:,:] = Sm
    op3[3,:,:] = Sp
    # Put into a list -----------------------------
    mpo = [op1,op2,op3]
    return mpo
