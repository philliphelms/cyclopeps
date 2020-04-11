"""
Tools for Projected Entangled Pair States

Author: Phillip Helms <phelms@caltech.edu>
Date: July 2019

.. Note, peps tensors are stored in order:

          (5) top
             |
(1) left  ___|___ (4) right
             |\
             | \
    (2) bottom  (3) physical
"""

from cyclopeps.tools.gen_ten import rand,einsum,eye,ones,svd_ten,zeros
#from cyclopeps.tools.params import *
from symtensor.settings import load_lib
from cyclopeps.tools.utils import *
from cyclopeps.tools.mps_tools import MPS,identity_mps
from numpy import float_
import numpy as np
import copy
FLIP = {'+':'-','-':'+'}

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PEPS ENVIRONMENT FUNCTIONS
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def copy_tensor_list(ten_list):
    """
    Create a copy of a list of tensors
    """
    ten_list_cp = [None]*len(ten_list)
    for i in range(len(ten_list)):
        ten_list_cp[i] = ten_list[i].copy()
    return ten_list_cp

def init_left_bmpo_sl(bra, ket=None, chi=4, truncate=True):
    """
    Create the initial boundary mpo for a peps

    Args:
        bra : list
            A list containing the tensors for a single peps column

    Kwargs:
        chi : int
            The maximum bond dimension for the boundary mpo
        truncate : bool
            Whether or not to do an svd and truncate the resulting
            boundary mpo
        ket : PEPS Object
            A second peps column, to use as the ket
            If None, then the bra col will be used

    Returns:
        bound_mpo : list
            An updated boundary mpo
    """
    mpiprint(3,'Initial Layer of left boundary mpo (sl)')

    # Find size of peps column and dims of tensors
    Ny = len(bra)

    # Copy the ket column if needed
    bra = copy_tensor_list(bra)
    if ket is None:
        ket = copy_tensor_list(bra)

    # Make list to hold resulting mpo
    bound_mpo = []

    for row in range(Ny):
        # Remove l and L empty indices
        ket[row] = ket[row].remove_empty_ind(0)
        bra[row] = bra[row].remove_empty_ind(0)
        # Add Bra-ket contraction
        res = einsum('dpru,DpRU->dDRurU',ket[row],bra[row])
        ressgn = res.get_signs()
        resleg = res.legs
        # Merge inds to make it an MPO
        res.merge_inds([0,1])
        res.merge_inds([2,3,4])
        # Append to boundary_mpo
        bound_mpo.append(res)

        # Add correct identity
        Dr = ket[row].shape[ket[row].legs[2][0]]
        Du = ket[row].shape[ket[row].legs[3][0]]
        Zr = ket[row].qn_sectors[ket[row].legs[2][0]]
        Zu = ket[row].qn_sectors[ket[row].legs[3][0]]
        I1 = eye(Dr,
                 Zr,
                 is_symmetric=ket[row].is_symmetric,
                 backend=ket[row].backend)
        I2 = eye(Du,
                 Zu,
                 is_symmetric=ket[row].is_symmetric,
                 backend=ket[row].backend)
        I3 = eye(Du,
                 Zu,
                 is_symmetric=ket[row].is_symmetric,
                 backend=ket[row].backend)
        # Make sure signs are correct
        if ressgn is not None:
            if ''.join(ressgn[i] for i in resleg[4]) == ''.join(I1.get_signs()[i] for i in I1.legs[0]): 
                I1.flip_signs()
            if ''.join(ressgn[i] for i in resleg[5]) == ''.join(I2.get_signs()[i] for i in I2.legs[0]): 
                I2.flip_signs()
            if ''.join(ressgn[i] for i in resleg[3]) == ''.join(I3.get_signs()[i] for i in I3.legs[0]): 
                I3.flip_signs()
        # Contract to form Identity
        Itmp = einsum('du,DU->dDuU',I3,I2)
        I = einsum('dDuU,lr->dlDruU',Itmp,I1)
        # Merge inds to make it an MPO
        I.merge_inds([0,1,2])
        I.merge_inds([2,3])
        # Append to the boundary mpo
        bound_mpo.append(I)

    # Put result into an MPS -------------------------------------------
    bound_mps = MPS(bound_mpo)

    # Reduce bond dimension
    if truncate:
        mpiprint(5,'Truncating Boundary MPS')
        if DEBUG:
            mpiprint(6,'Computing initial bmpo norm')
            norm0 = bound_mps.norm()
        bound_mps = bound_mps.apply_svd(chi)
        if DEBUG:
            mpiprint(6,'Computing resulting bmpo norm')
            norm1 = bound_mps.norm()
            mpiprint(4,'Norm Difference for chi={}: {}'.format(chi,abs(norm0-norm1)/abs(norm0)))

    return bound_mps

def left_bmpo_sl_add_ket(ket,bound_mpo,Ny,chi=4,truncate=True):
    """
    Add the ket layer to the boundary mpo
    """
    mpiprint(4,'Adding Ket')

    # Make list to hold resulting mpo
    bound_mpo_new = []

    for row in range(Ny):
        mpiprint(5,'Adding Site {} to Ket'.format(row))

        # Calculate ket contraction first (so we can use it to determine symmetry signs of identity)
        res = einsum('mln,ldpru->mdrpnu',bound_mpo[2*row+1],ket[row])
        ressgn = res.get_signs()
        resleg = res.legs
        # Reshape it into an MPO
        if row == Ny-1:
            res = res.remove_empty_ind(len(res.legs)-1)
            res.merge_inds([0,1])
            res.merge_inds([1,2])
        else:
            res.merge_inds([0,1])
            res.merge_inds([1,2])
            res.merge_inds([2,3])

        # Create Correct Identity
        Dd = ket[row].shape[ket[row].legs[1][0]]
        Zd = ket[row].qn_sectors[ket[row].legs[1][0]]
        I1 = eye(Dd,
                 Zd,
                 is_symmetric=ket[row].is_symmetric,
                 backend=ket[row].backend)
        # Adjust symmetry signs
        if ressgn is not None:
            if ''.join(ressgn[i] for i in resleg[0]) == ''.join(ressgn[i] for i in resleg[1]):
                I1.update_signs(''.join(FLIP[bound_mpo[2*row].get_signs()[i]] for i in bound_mpo[2*row].legs[2]) + 
                                ''.join(bound_mpo[2*row].get_signs()[i] for i in bound_mpo[2*row].legs[2]))
            else:
                I1.update_signs(''.join(bound_mpo[2*row].get_signs()[i] for i in bound_mpo[2*row].legs[2]) + 
                                ''.join(FLIP[bound_mpo[2*row].get_signs()[i]] for i in bound_mpo[2*row].legs[2]))
        # Contract with previous bmpo
        I = einsum('mLn,du->mdLnu',bound_mpo[2*row],I1)
        # Reshape it into an MPO
        I.merge_inds([0,1])
        I.merge_inds([2,3])

        # Append identity to boundary MPO
        bound_mpo_new.append(I)

        # Append ket to boundary MPO
        bound_mpo_new.append(res)

    # Put result into an MPS -------------------------------------------
    bound_mps = MPS(bound_mpo_new)

    # Reduce bond dimension
    if truncate:
        mpiprint(5,'Truncating Boundary MPS')
        if DEBUG:
            mpiprint(6,'Computing initial bmpo norm')
            norm0 = bound_mps.norm()
        bound_mps = bound_mps.apply_svd(chi)
        if DEBUG:
            mpiprint(6,'Computing resulting bmpo norm')
            norm1 = bound_mps.norm()
            mpiprint(4,'Norm Difference for chi={}: {}'.format(chi,abs(norm0-norm1)/abs(norm0)))

    return bound_mps

def left_bmpo_sl_add_bra(bra,bound_mpo,Ny,chi=4,truncate=True):
    """
    Add the bra layer to the boundary mpo
    """
    mpiprint(4,'Adding Bra')
    # Make list to hold resulting mpo
    bound_mpo_new = []

    for row in range(Ny):

        # Add bra contraction
        res = einsum('mLn,LDPRU->mDRnUP',bound_mpo[2*row],bra[row])
        # Save some useful info
        ressgn = res.get_signs()
        resleg = res.legs
        # Reshape it into an MPO
        if row == 0:
            res = res.remove_empty_ind(0)
            res.merge_inds([2,3,4])
        else:
            res.merge_inds([0,1])
            res.merge_inds([2,3,4])
        # Append to new boundary MPO
        bound_mpo_new.append(res)

        # Add correct identity
        mpiprint(6,'Adding Identity to boundary mps')
        # Unmerge bmps tensor
        bound_tens = bound_mpo[2*row+1]
        thermal = (len(bound_tens.legs[1]) == 3)
        bound_tens.unmerge_ind(1)
        if thermal: bound_tens.merge_inds([2,3])
        # Create identity tensor
        Du = bra[row].shape[bra[row].legs[4][0]]
        Zu = bra[row].qn_sectors[bra[row].legs[4][0]]
        I1 = eye(Du,
                Zu,
                is_symmetric=bra[row].is_symmetric,
                backend=bra[row].backend)
        # Adjust symmetry signs
        if ressgn is not None:
            if ''.join(ressgn[i] for i in resleg[3]) == ''.join(ressgn[i] for i in resleg[4]):
                I1.update_signs(''.join(bound_tens.get_signs()[i] for i in bound_tens.legs[0]) + 
                                ''.join(FLIP[bound_tens.get_signs()[i]] for i in bound_tens.legs[0]))
            else:
                I1.update_signs(''.join(FLIP[bound_tens.get_signs()[i]] for i in bound_tens.legs[0]) + 
                                ''.join(bound_tens.get_signs()[i] for i in bound_tens.legs[0]))
        # Contract with previous bmpo
        I = einsum('mrPn,DU->mDPrnU',bound_tens,I1)
        # Reshape into an MPO
        if row == Ny-1:
            I = I.remove_empty_ind(len(I.legs)-1)
            I.merge_inds([0,1,2])
        else:
            I.merge_inds([0,1,2])
            I.merge_inds([2,3])
        # Append to new boundary MPO
        bound_mpo_new.append(I)

    # Put result into an MPS -------------------------------------------
    bound_mps = MPS(bound_mpo_new)

    # Reduce bond dimension
    if truncate:
        mpiprint(5,'Truncating Boundary MPS')
        if DEBUG:
            mpiprint(6,'Computing initial bmpo norm')
            norm0 = bound_mps.norm()
        bound_mps = bound_mps.apply_svd(chi)
        if DEBUG:
            mpiprint(6,'Computing resulting bmpo norm')
            norm1 = bound_mps.norm()
            mpiprint(4,'Norm Difference for chi={}: {}'.format(chi,abs(norm0-norm1)/abs(norm0)))

    return bound_mps

def left_bmpo_sl(bra, bound_mpo, chi=4,truncate=True,ket=None):
    """
    Add two layers to the single layer boundary mpo environment

    Args:
        bra : list
            A list containing the tensors for a single peps column
        bound_mpo : list
            A list containing the tensors for the left neighboring
            boundary mpo

    Kwargs:
        chi : int
            The maximum bond dimension for the boundary mpo
        truncate : bool
            Whether or not to do an svd and truncate the resulting
            boundary mpo
        ket : PEPS Object
            A second peps column, to use as the ket
            If None, then the bra col will be used

    Returns:
        bound_mpo : list
            An updated boundary mpo
    """
    mpiprint(3,'Updating boundary mpo (sl)')
    # Find size of peps column and dims of tensors
    Ny = len(bra)

    # Copy the ket column if needed
    bra = copy_tensor_list(bra)
    if ket is None:
        ket = copy_tensor_list(bra)

    # First Layer (ket) #####################################
    bound_mpo = left_bmpo_sl_add_ket(ket,bound_mpo,Ny,chi=chi,truncate=truncate)
    # Second Layer (bra) ####################################
    bound_mpo = left_bmpo_sl_add_bra(bra,bound_mpo,Ny,chi=chi,truncate=truncate)

    # Return result
    return bound_mpo

def left_update_sl(peps_col, bound_mpo, chi=4,truncate=True,ket=None):
    """
    Update the boundary mpo, from the left, moving right, using single layer

    Args:
        peps_col : list
            A list containing the tensors for a single peps column
        bound_mpo : list
            The neighboring boundary mpo, which will be updated

    Kwargs:
        chi : int
            The maximum bond dimension for the boundary mpo
        truncate : bool
            Whether or not to do an svd and truncate the resulting
            boundary mpo
        ket : PEPS Object
            A second peps column, to use as the ket

    Returns:
        bound_mpo : list
            An updated boundary mpo
    """
    # Check if we are at left edge
    if bound_mpo is None:
        bound_mpo = init_left_bmpo_sl(peps_col,chi=chi,truncate=truncate,ket=ket)
    # Otherwise update is generic
    else:
        # Start from bottom of the column
        bound_mpo = left_bmpo_sl(peps_col,bound_mpo,chi=chi,truncate=truncate,ket=ket)
    return bound_mpo

def left_update(peps_col,bound_mpo,chi=4,ket=None):
    mpiprint(0,'Only single layer environment implemented')
    raise NotImplemented

def update_left_bound_mpo(peps_col, bound_mpo, chi=4, singleLayer=True,truncate=True,ket_col=None):
    """
    Update the boundary mpo, from the left, moving right

    Args:
        peps_col : list
            A list containing the tensors for a single peps column
        bound_mpo : list
            The neighboring boundary mpo, which will be updated

    Kwargs:
        chi : int
            The maximum bond dimension for the boundary mpo
        singleLayer : bool
            Indicates whether to use a single layer environment
            (currently it is the only option...)
        truncate : bool
            Whether or not to do an svd and truncate the resulting
            boundary mpo
        ket_col : PEPS Object
            A second peps column, to use as the ket

    Returns:
        bound_mpo : list
            An updated boundary mpo
    """
    if singleLayer:
        return left_update_sl(peps_col,bound_mpo,chi=chi,truncate=truncate,ket=ket_col)
    else:
        return left_update(peps_col,bound_mpo,chi=chi,truncate=truncate,ket=ket_col)

def calc_left_bound_mpo(peps,col,chi=4,singleLayer=True,truncate=True,return_all=False,ket=None):
    """
    Calculate the left boundary MPO

    Args:
        peps : List
            A list of lists containing the peps tensors
        col : int
            The last column for which you need the environment

    Kwargs:
        chi : int
            The maximum bond dimension of the boundary MPO
        single_layer : bool
            Indicates whether to use a single layer environment
            (currently it is the only option...)
        truncate : bool
            Whether or not to do an svd and truncate the resulting
            boundary mpo
        return_all : bool
            Whether to return a list of boundary mpos upto col or just
            return the boundary mpo for col.
        ket : PEPS Object
            A second peps, to use as the ket, in the operator contraction

    returns:
        bound_mpo : list
            An mpo stored as a list, corresponding to the
            resulting boundary mpo.

    """
    mpiprint(2,'Computing Left boundary MPO')
    # Determine the dimensions of the peps
    Nx = len(peps)
    Ny = len(peps[0])

    # Loop through the columns, creating a boundary mpo for each
    bound_mpo = [None]*(col-1)
    for colind in range(col-1):
        mpiprint(4,'Updating left boundary mpo')
        if ket is not None:
            ket_col = ket[colind][:]
        else: ket_col = None
        if colind == 0:
            bound_mpo[colind] = update_left_bound_mpo(peps[colind][:], None, chi=chi, singleLayer=singleLayer,truncate=truncate,ket_col=ket_col)
        else:
            bound_mpo[colind] = update_left_bound_mpo(peps[colind][:], bound_mpo[colind-1], chi=chi, singleLayer=singleLayer,truncate=truncate,ket_col=ket_col)

    # Return result
    if return_all:
        return bound_mpo
    else:
        return bound_mpo[-1]

def calc_right_bound_mpo(peps,col,chi=4,singleLayer=True,truncate=True,return_all=False,ket=None):
    """
    Calculate the right boundary MPO

    Args:
        peps : List
            A list of lists containing the peps tensors
        col : int or list of ints
            The column(s) for which you need the environment

    Kwargs:
        chi : int
            The maximum bond dimension of the boundary MPO
        single_layer : bool
            Indicates whether to use a single layer environment
            (currently it is the only option...)
        truncate : bool
            Whether or not to do an svd and truncate the resulting
            boundary mpo
        return_all : bool
            Whether to return a list of boundary mpos upto col or just
            return the boundary mpo for col.
        ket : PEPS Object
            A second peps, to use as the ket, in the operator contraction

    returns:
        bound_mpo : list
            An mpo stored as a list, corresponding to the
            resulting boundary mpo.

    """
    mpiprint(2,'Computing Left boundary MPO')

    # Determine the dimensions of the peps
    Nx = len(peps)
    Ny = len(peps[0])

    # Flip the peps
    peps = flip_peps(peps)
    if ket is not None:
        ket = flip_peps(ket)
    col = Nx-col

    # Loop through the columns, creating a boundary mpo for each
    bound_mpo = [None]*(col-1)
    for colind in range(col-1):
        mpiprint(4,'Updating boundary mpo')
        if ket is not None:
            ket_col = ket[colind][:]
        else: ket_col = None
        if colind == 0:
            bound_mpo[colind] = update_left_bound_mpo(peps[colind][:], None, chi=chi, singleLayer=singleLayer, truncate=truncate, ket_col=ket_col)
        else:
            bound_mpo[colind] = update_left_bound_mpo(peps[colind][:], bound_mpo[colind-1], chi=chi, singleLayer=singleLayer, truncate=truncate, ket_col=ket_col)

    # Unflip the peps
    peps = flip_peps(peps)
    if ket is not None:
        ket = flip_peps(ket)

    # Return results
    if return_all:
        return bound_mpo[::-1]
    else:
        return bound_mpo[-1]

def rotate_peps(peps,clockwise=True):
    """
    Rotate a peps

    Args:
        peps : a list of a list containing peps tensors
            The initial peps tensor

    Kwargs:
        clockwise : bool
            Rotates clockwise if True, counter-clockwise
            otherwise

    Returns:
        peps : a list of a list containing peps tensors
            The horizontally flipped version of the peps
            tensor. This is flipped such that ...
    """

    # Get system size
    Nx = len(peps)
    Ny = len(peps[0])

    # Create empty peps
    rpeps = []
    for y in range(Ny):
        tmp = []
        for x in range(Nx):
            tmp += [None]
        rpeps += [tmp]

    # Copy peps, but rotated
    for x in range(Nx):
        for y in range(Ny):
            if clockwise:
                # Copy Correct Tensor
                rpeps[y][Nx-1-x] = peps[x][y].copy()
                # Reorder Indices
                rpeps[y][Nx-1-x] = rpeps[y][Nx-1-x].transpose([1,3,2,4,0])
            else:
                # Copy Correct Tensor
                rpeps[Ny-1-y][x] = peps[x][y].copy()
                # Reorder Indices
                rpeps[Ny-1-y][x] = rpeps[Ny-1-y][x].transpose([4,0,2,1,3])

    # Return Rotated peps
    return rpeps

def rotate_lambda(Lambda,clockwise=True):
    """
    Rotate the Lambda tensors for the canonical PEPS representation
    """
    if Lambda is not None:

        # Get system size (of rotated lambda)
        Ny = len(Lambda[0])
        Nx = len(Lambda[1][0])

        # Lambda tensors along vertical bonds
        vert = []
        for x in range(Nx):
            tmp = []
            for y in range(Ny-1):
                if clockwise:
                    tmp += [Lambda[1][Ny-2-y][x].copy()]
                else:
                    tmp += [Lambda[1][y][Nx-1-x].copy()]
            vert += [tmp]

        # Lambda tensors along horizontal bonds
        horz = []
        for x in range(Nx-1):
            tmp = []
            for y in range(Ny):
                if clockwise:
                    tmp += [Lambda[0][Ny-1-y][x].copy()]
                else:
                    tmp += [Lambda[0][y][Nx-2-x].copy()]
            horz += [tmp]

        # Combine vertical and horizontal lambdas
        rLambda = [vert,horz]
        return rLambda
    else:
        return None

def flip_peps(peps,mk_copy=True):
    """
    Flip a peps horizontally

    Args:
        peps : a list of a list containing peps tensors
            The initial peps tensor

    Kwargs:
        mk_copy : bool
            Whether to make this a copy of the original peps

    Returns:
        peps : a list of a list containing peps tensors
            The horizontally flipped version of the peps
            tensor. This is a copy of the original peps
    """

    # Get system size
    Nx = len(peps)
    Ny = len(peps[0])

    # Create empty peps
    fpeps = []
    for x in range(Nx):
        tmp = []
        for y in range(Ny):
            tmp += [None]
        fpeps += [tmp]

    # Copy peps, but flipped
    for x in range(Nx):
        for y in range(Ny):
            # Copy Correct Tensor
            fpeps[x][y] = peps[(Nx-1)-x][y].copy()
            # Reorder Indices
            fpeps[x][y] = fpeps[x][y].transpose([3,1,2,0,4])

    # Return Flipped peps
    return fpeps

def flip_lambda(Lambda):
    """
    Flip the lambda tensors (part of the canonical peps) horizontally

    Args:
        Lambda :

    Returns:
        Lambda :
            The horizontally flipped version of the lambda
            tensor. This is flipped such that ...
    """

    if Lambda is not None:
        # Get system size
        Nx = len(Lambda[0])
        Ny = len(Lambda[1][0])

        # Lambda tensors along vertical bonds
        vert = []
        for x in range(Nx):
            tmp = []
            for y in range(Ny-1):
                tmp += [Lambda[0][(Nx-1)-x][y].copy()]
            vert += [tmp]
        # Lambda tensors along horizontal bonds
        horz = []
        for x in range(Nx-1):
            tmp = []
            for y in range(Ny):
                tmp += [Lambda[1][(Nx-2)-x][y].copy()]
            horz += [tmp]

        # Add to tensors
        fLambda = [vert,horz]

        # Return Flipped peps
        return fLambda
    else:
        return None

def peps_col_to_mps(peps_col):
    """
    Convert a PEPS column into an MPS.
    The structure of the resulting MPS tensors is:

        left/phys/right
              |
              |
    bottom ---+--- top


    Args:
        peps_col : 1D Array
            A list containing the tensors for each site in a peps column

    Returns:
        mps : 1D Array
            The resulting 1D array containing the PEPS column's tensor

    """

    # Determine number of rows
    Ny = len(peps_col)

    # Create a list to hold the copy
    peps_col_cp = [None]*Ny
    for i in range(len(peps_col)):
        peps_col_cp[i] = peps_col[i].copy()
    peps_col = peps_col_cp

    for row in range(Ny):

        # Transpose to put left, physical, and right bonds in middle
        peps_col[row] = peps_col[row].transpose([1,0,2,3,4])
        # lump left, physical, and right tensors
        peps_col[row].merge_inds([1,2,3])

    # Convert PEPS column into an MPS
    mps = MPS(peps_col)

    # Return resulting mps
    return mps

def calc_peps_col_norm(peps_col):
    """
    Convert a PEPS column into an MPS, then take the norm of that MPS
    .. Note : Used to keep PEPS norm near 1.

    Args:
        peps_col : 1D Array
            A list containing the tensors for each site in a peps column

    Returns:
        norm : float
            The norm of the peps column (reshaped as an MPS)
    """

    # Convert peps column to an mps by lumping indices
    mps = peps_col_to_mps(peps_col)

    # Compute the norm of that mps
    norm = 0.5*mps.norm()

    # Return the resulting norm
    return norm

def thermal_peps_tensor(Nx,Ny,x,y,d,D,Zn=None,dZn=None,backend='numpy',dtype=float_):
    """
    Create a thermal (beta=0) tensor for a PEPS

    Args:
        Nx : int
            The PEPS lattice size in the x-direction
        Ny : int
            The PEPS lattice size in the y-direction
        x : int
            The x-coordinate of the tensor
        y : int
            The y-coordinate of the tensor

    Kwargs:
        Zn : int
            Create a PEPS which preserves this Zn symmetry,
            i.e. if Zn=2, then Z2 symmetry is preserved.
        dZn : int
            The number of symmetry sectors for the physical bond dimension
            if None, then Zn will be used
        backend : str
            This specifies the backend to be used for the calculation.
            Options are currently 'numpy' or 'ctf'. If using symmetries,
            this will be adapted to using symtensors with numpy or ctf as
            the backend.
        dtype : dtype
            The data type of the tensor
            Default : np.float_

    Returns:
        ten : ndarray
            A random tensor with the correct dimensions
            for the given site
    """
    # Determine the correct bond dimensions
    Dl = D
    Dr = D
    Du = D
    Dd = D

    # Set to one if at an edge
    if x == 0:    Dl = 1
    if x == Nx-1: Dr = 1
    if y == 0:    Dd = 1
    if y == Ny-1: Du = 1

    # Set default value of sym
    sym = None

    # Deal with Zn symmetry (if needed)
    if Zn is not None:
        # And correct symmetries
        Znl= Zn
        Znr= Zn
        Znu= Zn
        Znd= Zn
        # Set to one if at an edge
        if x == 0:    Znl = 1
        if x == Nx-1: Znr = 1
        if y == 0:    Znd = 1
        if y == Ny-1: Znu = 1
        # Resize D->Dnew so Dnew*Zn = D
        Dl /= Znl
        Dr /= Znr
        Dd /= Znd
        Du /= Znu
        d  /= dZn

        # Create sym argument
        sym = ['+++---',
               [range(Znl),range(Znd),range(dZn),range(dZn),range(Znr),range(Znu)],
               0,
               Zn]

    # Create an empty tensor
    dims = (Dl,Dd,d,d,Dr,Du)
    ten = zeros(dims,sym,backend=backend,dtype=dtype,legs=[[0],[1],[2,3],[4],[5]])
    
    # Fill tensor entries where needed
    if sym is None:
        for i in range(d):
            for j in range(d):
                if i == j:
                    ten.ten[0,0,i,j,0,0] = 1./ten.backend.sqrt(float(d))
    else:
        for i in range(d):
            for j in range(d):
                for k in range(dZn):
                    for l in range(dZn):
                        if (i == j) and (k == l):
                            ten.ten.array[0,0,k,l,0,0,0,i,j,0,0] = 1./ten.backend.sqrt(float(d))

    # Return result
    return ten

def rand_peps_tensor(Nx,Ny,x,y,d,D,Zn=None,dZn=None,backend='numpy',dtype=float_):
    """
    Create a random tensor for a PEPS

    Args:
        Nx : int
            The PEPS lattice size in the x-direction
        Ny : int
            The PEPS lattice size in the y-direction
        x : int
            The x-coordinate of the tensor
        y : int
            The y-coordinate of the tensor

    Kwargs:
        Zn : int
            Create a PEPS which preserves this Zn symmetry,
            i.e. if Zn=2, then Z2 symmetry is preserved.
        dZn : int
            The number of symmetry sectors for the physical bond dimension
            if None, then Zn will be used
        backend : str
            This specifies the backend to be used for the calculation.
            Options are currently 'numpy' or 'ctf'. If using symmetries,
            this will be adapted to using symtensors with numpy or ctf as
            the backend.
        dtype : dtype
            The data type of the tensor
            Default : np.float_

    Returns:
        ten : ndarray
            A random tensor with the correct dimensions
            for the given site
    """
    # Determine the correct bond dimensions
    Dl = D
    Dr = D
    Du = D
    Dd = D

    # Set to one if at an edge
    if x == 0:    Dl = 1
    if x == Nx-1: Dr = 1
    if y == 0:    Dd = 1
    if y == Ny-1: Du = 1

    # Set default value of sym
    sym = None

    # Deal with Zn symmetry (if needed)
    if Zn is not None:
        # And correct symmetries
        Znl= Zn
        Znr= Zn
        Znu= Zn
        Znd= Zn
        # Set to one if at an edge
        if x == 0:    Znl = 1
        if x == Nx-1: Znr = 1
        if y == 0:    Znd = 1
        if y == Ny-1: Znu = 1
        # Resize D->Dnew so Dnew*Zn = D
        Dl /= Znl
        Dr /= Znr
        Dd /= Znd
        Du /= Znu
        d  /= dZn

        # Create sym argument
        sym = ['+++--',
               [range(Znl),range(Znd),range(dZn),range(Znr),range(Znu)],
               0,
               Zn]

    # Create the random tensor
    dims = (Dl,Dd,d,Dr,Du)
    ten = rand(dims,sym,backend=backend,dtype=dtype)
    #ten = 0.95*ones(dims,sym,backend=backend,dtype=dtype) + 0.1*rand(dims,sym,backend=backend,dtype=dtype)
    
    # Return result
    return ten

def normalize_peps_col(peps_col):
    """
    Try to keep the norm of a PEPS column near 1.

    Args:
        peps_col : 1D Array
            A list containing the tensors for each site in a peps column

    Returns:
        peps_col : 1D Array
            A normalized version of the input peps_col

    """

    # Figure out column height
    Ny = len(peps_col)

    # Compute the norm
    norm = calc_peps_col_norm(peps_col)

    # Normalize each of the tensors
    for row in range(Ny):
        peps_col[row] *= 1. / (norm ** (0.5 / Ny) )

    # Return the normalized peps column
    return peps_col

def multiply_peps_elements(peps,const):
    """
    Multiply all elements in a peps by a constant

    Args:
        peps : A PEPS object or a list of lists containing the peps tensors

        const : float
            The constant with which to multiply each peps tensor

    Returns:
        peps : a PEPS object, or list of lists, depending on input
    """

    Nx = len(peps)
    Ny = len(peps[0])
    for xind in range(Nx):
        for yind in range(Ny):
            peps[xind][yind] *= const
    return peps

def normalize_peps(peps,max_iter=100,norm_tol=20,chi=4,up=100.0,
                    down=0.0,singleLayer=True):
    """
    Normalize the full PEPS by doing a binary search on the
    interval [down, up] for the factor which, when multiplying
    every element of the PEPS tensors, yields a rescaled PEPS
    with norm equal to 1.0.

    Args:
        peps : A PEPS object
            The PEPS to be normalized, given as a PEPS object

    Kwargs:
        max_iter : int
            The maximum number of iterations of the normalization
            procedure. Default is 20.
        norm_tol : int
            We require the measured norm to be within the bounds
            10^(-norm_tol) < norm < 10^(norm_tol) before we do
            exact arithmetic to get the norm very close to 1. Default
            is 20.
        chi : int
            Boundary MPO maximum bond dimension
        up : float
            The upper bound for the binary search factor. Default is 1.0,
            which assumes that the norm of the initial PEPS is greater
            than 10^(-norm_tol) (this is almost always true).
        down : float
            The lower bound for the binary search factor. Default is 0.0.
            The intial guess for the scale factor is the midpoint
            between up and down. It's not recommended to adjust the
            up and down parameters unless you really understand what
            they are doing.
        single_layer : bool
            Indicates whether to use a single layer environment
            (currently it is the only option...)

    Returns:
        norm : float
            The approximate norm of the PEPS after the normalization
            procedure
        peps : list
            The normalized version of the PEPS, given as a PEPS object

    """

    # Figure out peps size
    Nx = peps.Nx
    Ny = peps.Ny
    be = peps[0][0].backend

    pwr = -1.0 / (2*Nx*Ny) # NOTE: if trying to use this procedure to
                           # normalize a partition function, remove
                           # the factor of 2 in this denominator
    mpiprint(4, '\n[binarySearch] shape=({},{}), chi={}'.format(Nx,Ny,chi))

    # Check if state is already easily normalized
    try:
        z = calc_peps_norm(peps,chi=chi,singleLayer=singleLayer)
    except:
        z = None
    if not (z < 10.**(-1*norm_tol) or z > 10.**(norm_tol)) or (z is None):
        if z is not None:
            sfac = z**pwr
            peps_try = multiply_peps_elements(peps.copy(),sfac)
            z = calc_peps_norm(peps_try,chi=chi,singleLayer=singleLayer)
        else:
            try:
                sfac = 1e-3
                peps_try = multiply_peps_elements(peps.copy(),sfac)
                z = calc_peps_norm(peps_try,chi=chi,singleLayer=singleLayer)
            except:
                sfac = 1e3
                peps_try = multiply_peps_elements(peps.copy(),sfac)
                z = calc_peps_norm(peps_try,chi=chi,singleLayer=singleLayer)
        if abs(z-1.) < 1e-6: 
            return z, peps_try

    # get initial scale factor
    scale = (up+down)/2.0

    # begin search
    peps_try = multiply_peps_elements(peps.copy(),scale)

    istep = 0
    while True:
        try:
            istep += 1
            z = None
            z = calc_peps_norm(peps_try,chi=chi,singleLayer=singleLayer)
        except:
            print('Failed to calculate peps norm')
            pass
        mpiprint(2, 'step={}, (down,up)=({},{}), scale={}, norm={}'.format(
                                                        istep,down,up,scale,z))
        # if an exception is thrown in calc_peps_norm because scale is too large
        if z == None:
            up = scale
            scale = scale / 2.0
        # adjust scale to make z into target region
        else:
            if abs(z-1.0) < 1e-6:
                mpiprint(2, 'converged scale = {}, norm = {}'.format(scale,z))
                break
            if z < 10.0**(-1*norm_tol) or z > 10.0**(norm_tol) or be.isnan(z):
                if z > 1.0 or be.isnan(z):
                    up = scale
                    scale = (up+down)/2.0
                else:
                    down = scale
                    scale = (up+down)/2.0
            # close to convergence, apply "exact" scale
            else:
                sfac = z**pwr
                scale = sfac*scale
                mpiprint(2, 'apply exact scale: {}'.format(scale))

        if istep == max_iter:
            mpiprint(4, 'binarySearch normalization exceeds max_iter... terminating')
            break

        peps_try = multiply_peps_elements(peps.copy(),scale)

    return z, peps_try

def calc_peps_norm(_peps,chi=4,singleLayer=True,ket=None):
    """
    Calculate the norm of the PEPS

    Args:
        peps : A PEPS object
            The PEPS for which we will compute the norm

    Kwargs:
        chi : int
            The boundary MPO bond dimension
        single_layer : bool
            Indicates whether to use a single layer environment
            (currently it is the only option...)

    Returns:
        norm : float
            The (approximate) norm of the PEPS
    """
    # Absorb Lambda tensors if needed
    if _peps.ltensors is not None:
        peps = _peps.copy()
        peps.absorb_lambdas()
    else:
        peps = _peps.copy()
    if ket is not None and ket.ltensors is not None:
        ket = ket.copy()
        ket.absorb_lambdas()
    elif ket is not None:
        ket = ket.copy()

    # Get PEPS Dims
    Nx = len(peps)
    Ny = len(peps[0])

    # Get the boundary MPO from the left (for the furthest right column)
    left_bound_mpo  = calc_left_bound_mpo(peps,Nx,chi=chi,singleLayer=singleLayer,ket=ket)

    # Get the boundary MPO from the right (for the furthest right column)
    right_bound_mpo = calc_right_bound_mpo(peps,Nx-2,chi=chi,singleLayer=singleLayer,ket=ket)

    # Contract the two MPOs
    norm = left_bound_mpo.contract(right_bound_mpo)

    # Return result
    return norm

def make_thermal_peps(Nx,Ny,d,D,Zn=None,dZn=None,backend='numpy',dtype=float_):
    """
    Make a thermal (beta=0) PEPS

    Args:
        d : int
            The local bond dimension
        D : int
            The auxilliary bond dimension
        Nx : int
            The PEPS lattice size in the x-direction
        Ny : int
            The PEPS lattice size in the y-direction

    Kwargs:
        Zn : int
            Create a PEPS which preserves this Zn symmetry,
            i.e. if Zn=2, then Z2 symmetry is preserved.
        dZn : int
            The number of symmetry sectors for the physical bond dimension
            If None, then will be the same as Zn
        backend : str
            This specifies the backend to be used for the calculation.
            Options are currently 'numpy' or 'ctf'. If using symmetries,
            this will be adapted to using symtensors with numpy or ctf as
            the backend.
        dtype : dtype
            The data type of the tensor
            Default : np.float_

    Returns:
        peps : array of arrays
            A random peps held as an array of arrays
    """
    # Create a list of lists to hold PEPS tensors
    tensors = []
    for x in range(Nx):
        tmp = []
        for y in range(Ny):
            tmp += [None]
        tensors += [tmp]

    # Place random tensors into the PEPS
    for x in range(Nx):
        for y in range(Ny):
            tensors[x][y] = thermal_peps_tensor(Nx,Ny,x,y,d,D,Zn=Zn,dZn=dZn,backend=backend,dtype=dtype)

    return tensors

def make_rand_peps(Nx,Ny,d,D,Zn=None,dZn=None,backend='numpy',dtype=float_):
    """
    Make a random PEPS

    Args:
        d : int
            The local bond dimension
        D : int
            The auxilliary bond dimension
        Nx : int
            The PEPS lattice size in the x-direction
        Ny : int
            The PEPS lattice size in the y-direction

    Kwargs:
        Zn : int
            Create a PEPS which preserves this Zn symmetry,
            i.e. if Zn=2, then Z2 symmetry is preserved.
        dZn : int
            The number of symmetry sectors for the physical bond dimension
            If None, then will be the same as Zn
        backend : str
            This specifies the backend to be used for the calculation.
            Options are currently 'numpy' or 'ctf'. If using symmetries,
            this will be adapted to using symtensors with numpy or ctf as
            the backend.
        dtype : dtype
            The data type of the tensor
            Default : np.float_

    Returns:
        peps : array of arrays
            A random peps held as an array of arrays
    """
    # Create a list of lists to hold PEPS tensors
    tensors = []
    for x in range(Nx):
        tmp = []
        for y in range(Ny):
            tmp += [None]
        tensors += [tmp]

    # Place random tensors into the PEPS
    for x in range(Nx):
        for y in range(Ny):
            tensors[x][y] = rand_peps_tensor(Nx,Ny,x,y,d,D,Zn=Zn,dZn=dZn,backend=backend,dtype=dtype)
        # At the end of each column, make the norm smaller
        tensors[x][:] = normalize_peps_col(tensors[x][:])

    return tensors

def thermal_lambda_tensor(D,Zn=None,backend='numpy',dtype=float_):
    """
    Create a thermal (currently identity) lambda tensor for a canonical PEPS

    Args:
        D : int
            The PEPS Bond Dimension

    Kwargs:
        Zn : int
            Create a PEPS which preserves this Zn symmetry,
            i.e. if Zn=2, then Z2 symmetry is preserved.
        backend : str
            This specifies the backend to be used for the calculation.
            Options are currently 'numpy' or 'ctf'. If using symmetries,
            this will be adapted to using symtensors with numpy or ctf as
            the backend.
        dtype : dtype
            The data type of the tensor
            Default : np.float_

    Returns:
        ten : ndarray
            A random tensor with the correct dimensions
            for the given site
    """
    # Determine symmetry
    sym = None
    if Zn is not None:
        sym = ['+-',[range(Zn)]*2,0,Zn]
        D /= Zn

    # Create empty tensor
    l = zeros((D,D),
              sym=sym,
              backend=backend,
              dtype=dtype)

    # Fill Diagonal Elements
    if l.sym is None:
        l.ten = l.backend.diag(l.backend.ones(D))
    else:
        for i in range(Zn):
            l.ten.array[i,:,:] = l.backend.diag(l.backend.ones(D))

    # Return result
    return l

def rand_lambda_tensor(D,Zn=None,backend='numpy',dtype=float_):
    """
    Create a random lambda tensor for a canonical PEPS

    Args:
        D : int
            The PEPS Bond Dimension

    Kwargs:
        Zn : int
            Create a PEPS which preserves this Zn symmetry,
            i.e. if Zn=2, then Z2 symmetry is preserved.
        backend : str
            This specifies the backend to be used for the calculation.
            Options are currently 'numpy' or 'ctf'. If using symmetries,
            this will be adapted to using symtensors with numpy or ctf as
            the backend.
        dtype : dtype
            The data type of the tensor
            Default : np.float_

    Returns:
        ten : ndarray
            A random tensor with the correct dimensions
            for the given site
    """
    # Determine symmetry
    sym = None
    if Zn is not None:
        sym = ['+-',[range(Zn)]*2,0,Zn]
        D /= Zn

    # Create empty tensor
    l = zeros((D,D),
              sym=sym,
              backend=backend,
              dtype=dtype)

    # Fill Diagonal Elements
    if l.sym is None:
        l.ten = l.backend.diag(l.backend.random(D))
    else:
        for i in range(Zn):
            l.ten.array[i,:,:] = l.backend.diag(l.backend.random(D))

    # Return result
    return l

def make_thermal_lambdas(Nx,Ny,D,Zn=None,backend='numpy',dtype=float_):
    """
    Make identites as diagonal matrices to serve as the
    singular values for the Gamma-Lambda canonical
    form of the thermal PEPS

    Used primarily for the simple update contraction scheme
    """

    # Lambda tensors along vertical bonds
    vert = []
    for x in range(Nx):
        tmp = []
        for y in range(Ny-1):
            tmp += [thermal_lambda_tensor(D,Zn=Zn,backend=backend,dtype=dtype)]
        vert += [tmp]

    # Lambda tensors along horizontal bonds
    horz = []
    for x in range(Nx-1):
        tmp = []
        for x in range(Ny):
            tmp += [thermal_lambda_tensor(D,Zn=Zn,backend=backend,dtype=dtype)]
        horz += [tmp]

    # Add horizontal and vertical lambdas to tensor list
    tensors = [vert,horz]
    return tensors

def make_rand_lambdas(Nx,Ny,D,Zn=None,backend='numpy',dtype=float_):
    """
    Make random diagonal matrices to serve as the
    singular values for the Gamma-Lambda canonical
    form of the PEPS

    Used primarily for the simple update contraction scheme
    """

    # Lambda tensors along vertical bonds
    vert = []
    for x in range(Nx):
        tmp = []
        for y in range(Ny-1):
            tmp += [rand_lambda_tensor(D,Zn=Zn,backend=backend,dtype=dtype)]
        vert += [tmp]

    # Lambda tensors along horizontal bonds
    horz = []
    for x in range(Nx-1):
        tmp = []
        for x in range(Ny):
            tmp += [rand_lambda_tensor(D,Zn=Zn,backend=backend,dtype=dtype)]
        horz += [tmp]

    # Add horizontal and vertical lambdas to tensor list
    tensors = [vert,horz]
    return tensors

def update_top_env_gen(row,bra,ket,left1,left2,right1,right2,prev_env,chi=10,truncate=True):
    """
    Doing the following contraction:

    +----+         +----+     +----+             +----+   
    | p1 |-- ...  -| p2 |-----| p3 |-   ...   ---| p6 |
    +----+         +----+     +----+             +----+   
       |              |          |                  |  
       a              b          c                  f  
       |              |          |                  |  
    +----+         +----+        |               +----+
    | l2 |-g ...  -| k1 |-----H--^--h   ...   ---| r2 |
    +----+         +----+        |               +----+  
       |              |  \       |                  |   
       |              |   \      |                  |   
       j              |    l     |                  q    
       |              |     \    |                  |  
       |              |      \   |                  |   
    +----+            |       +----+             +----+
    | l1 |-- ...  -R--^--r----| b1 |-s  ...   ---| r1 |
    +----+            |       +----+             +----+
       |              |          |                  |  
       |              |          |                  |  
       u              k          v                  x

    """
    # Determine if it is a thermal state
    thermal = len(bra[0][row].legs[2]) == 2
    
    # Figure out number of columns
    ncol = len(bra)

    # Create the new top environment
    if prev_env is None:
        # Create the first top environment
        top_env = []

        # First site is the current left bound_mpo
        res = einsum('urj,jga->augr',left1,left2)
        # Merge needed inds
        res.merge_inds([2,3])
        top_env.append(res)

        # Loop through and add bras and kets
        for col in range(ncol):
            # Copy the needed tensors
            ketten = ket[col][row].copy()
            braten = bra[col][row].copy()

            # Add ket -----------------------------------
            # Remove top ind
            ketten = ketten.remove_empty_ind(4) 
            # Create correct identity
            # TODO - Make sure signs are correct (will give error in symmetric case)
            Dl = braten.shape[braten.legs[0][0]]
            Zl = braten.qn_sectors[braten.legs[0][0]]
            I = eye(Dl,
                    Zl,
                    is_symmetric=braten.is_symmetric,
                    backend=braten.backend)
            if len(braten.legs[0]) > 1:
                for legind in range(1,len(braten.legs[0])):
                    Dli = braten.shape[braten.legs[0][legind]]
                    Zli = braten.qn_sectors[braten.legs[0][legind]]
                    Ii = eye(Dli,
                             Zli,
                             is_symmetric=braten.is_symmetric,
                             backend=braten.backend)
                    I = einsum('ij,IJ->iIjJ',I,Ii)
                    I.merge_inds([0,1])
                    I.merge_inds([1,2])
            # Contract identity with the ket
            res = einsum('gklh,Rr->gRkhlr',ketten,I)
            # Merge Correct inds
            res.merge_inds([0,1])
            res.merge_inds([2,3,4])
            # Add to top_env
            top_env.append(res)

            # Add bra ----------------------------------
            # Remove top ind
            braten = braten.remove_empty_ind(4)
            # Create correct identity
            # TODO - Make sure signs are correct (will give error in symmetric case)
            Dl = ketten.shape[ketten.legs[3][0]]
            Zl = ketten.qn_sectors[ketten.legs[3][0]]
            I = eye(Dl,
                    Zl,
                    is_symmetric=ketten.is_symmetric,
                    backend=ketten.backend)
            if len(ketten.legs[3]) > 1:
                for legind in range(1,len(ketten.legs[3])):
                    Dli = ketten.shape[ketten.legs[3][legind]]
                    Zli = ketten.qn_sectors[ketten.legs[3][legind]]
                    Ii = eye(Dli,
                             Zli,
                             is_symmetric=ketten.is_symmetric,
                             backend=ketten.backend)
                    I = einsum('ij,IJ->iIjJ',I,Ii)
                    I.merge_inds([0,1])
                    I.merge_inds([1,2])
            # Contract identity with the ket
            res = einsum('rvls,Hh->Hlrvhs',braten,I)
            # Merge Correct inds
            res.merge_inds([0,1,2])
            res.merge_inds([2,3])
            # Add to top_env
            top_env.append(res)

        # Last site is the current right bound_mpo
        res = einsum('xsq,qhf->hsxf',right1,right2)
        # Merge needed inds
        res.merge_inds([0,1])
        top_env.append(res)

        # Put result into an MPS -------------------------------------------
        top_env = MPS(top_env)

        # Reduce bond dimension
        if truncate:
            mpiprint(5,'Truncating Boundary MPS')
            if DEBUG:
                mpiprint(6,'Computing initial bmpo norm')
                norm0 = top_env.norm()
            top_env = top_env.apply_svd(chi)
            if DEBUG:
                mpiprint(6,'Computing resulting bmpo norm')
                norm1 = top_env.norm()
                mpiprint(0,'Norm Difference for chi={}: {}'.format(chi,abs(norm0-norm1)/abs(norm0)))
    else:
        # Add the ket layer --------------------------------------------------
        """
        Doing the following contraction:
           +----+   +----+     +----+           +----+    +----+   +----+   
        z--| p1 |-y-| p2 |--x--| p3 |-w  ...   -| p4 |-v--| p5 |-u-| p6 |--t
           +----+   +----+     +----+           +----+    +----+   +----+   
              |        |          |               |          |        |  
              a        b          c               d          e        f  
              |        |          |               |          |        |  
           +----+   +----+        |             +----+       |     +----+
           | l2 |-g-| k1 |-----h--^---   ...   -| k2 |----i--^-----| r2 |
           +----+   +----+-------+|             +----+------+|     +----+  
              |        |         ||               |         ||        |   
              |        |         ||               |         ||        |   
              j        k         lc               n         ow        q    
        """
        # Create the next top environment
        top_env = []
        # First absorb left boundary mpo
        res = einsum('jga,zay->zjyg',left2,prev_env[0])
        # Merge correct inds
        res.merge_inds([2,3])
        # Add to top_env
        top_env.append(res)
        
        # Loop through and add kets
        for col in range(ncol):
            # Add ket --------------------------
            ketten = ket[col][row].copy()
            # Contract with previous top env
            res = einsum('gklhb,ybx->ygkxhl',ketten,prev_env[2*col+1])
            # Merge correct indices
            res.merge_inds([0,1])
            res.merge_inds([2,3,4])
            # Add to top_env
            top_env.append(res)

            # Add identity ---------------------
            # TODO - Make sure signs are correct (will give error in symmetric case)
            D1 = ketten.shape[ketten.legs[3][0]]
            Z1 = ketten.qn_sectors[ketten.legs[3][0]]
            I1 = eye(D1,
                     Z1,
                     is_symmetric=ketten.is_symmetric,
                     backend=ketten.backend)
            if len(ketten.legs[3]) > 1:
                for legind in range(1,len(ketten.legs[3])):
                    Dli = ketten.shape[ketten.legs[3][legind]]
                    Zli = ketten.qn_sectors[ketten.legs[3][legind]]
                    Ii = eye(Dli,
                             Zli,
                             is_symmetric=ketten.is_symmetric,
                             backend=ketten.backend)
                    I1 = einsum('ij,IJ->iIjJ',I1,Ii)
                    I1.merge_inds([0,1])
                    I1.merge_inds([1,2])
            D2 = ketten.shape[ketten.legs[2][0]]
            Z2 = ketten.qn_sectors[ketten.legs[2][0]]
            I2 = eye(D2,
                     Z2,
                     is_symmetric=ketten.is_symmetric,
                     backend=ketten.backend)
            if len(ketten.legs[2]) > 1:
                for legind in range(1,len(ketten.legs[2])):
                    Dli = ketten.shape[ketten.legs[2][legind]]
                    Zli = ketten.qn_sectors[ketten.legs[2][legind]]
                    Ii = eye(Dli,
                             Zli,
                             is_symmetric=ketten.is_symmetric,
                             backend=ketten.backend)
                    I2 = einsum('ij,IJ->iIjJ',I2,Ii)
                    I2.merge_inds([0,1])
                    I2.merge_inds([1,2])
            # Contract with previous environment
            res = einsum('xcw,Hh->xHcwh',prev_env[2*col+2],I1)
            res = einsum('xHcwh,Ll->xHLlcwh',res,I2)
            # Merge correct indices
            res.merge_inds([0,1,2])
            res.merge_inds([1,2])
            res.merge_inds([2,3])
            # Add to top_env
            top_env.append(res)

        # Last, absorb right boundary mpo
        res = einsum('qif,uft->uiqt',right2,prev_env[2*ncol+1])
        # Merge needed inds
        res.merge_inds([0,1])
        # Add to top_env
        top_env.append(res)

        # Put result into an MPS ------------------
        top_env = MPS(top_env)

        # Reduce bond dimension
        if truncate:
            mpiprint(5,'Truncating Boundary MPS')
            if DEBUG:
                mpiprint(6,'Computing initial bmpo norm')
                norm0 = top_env.norm()
            top_env = top_env.apply_svd(chi)
            if DEBUG:
                mpiprint(6,'Computing resulting bmpo norm')
                norm1 = top_env.norm()
                mpiprint(0,'Norm Difference for chi={}: {}'.format(chi,abs(norm0-norm1)/abs(norm0)))

        # Update prev_env
        prev_env = top_env

        # Add the bra layer --------------------------------------------------
        """
        Doing the following contraction:
           +----+   +----+     +----+           +----+    +----+   +----+   
        z--| p1 |-y-| p2 |--x--| p3 |-w  ...   -| p4 |-v--| p5 |-u-| p6 |--g
           +----+   +----+     +----+           +----+    +----+   +----+   
              |        |         ||               |         ||        |  
              a        |         lc               d         oe        f  
              |        |         ||               |         ||        |  
           +----+      |       +----+             |       +----+   +----+
           | l1 |---r--^-------| b1 |-   ...   ---^-s-----| b2 |-t-| r1 |
           +----+      |       +----+             |       +----+   +----+
              |        |          |               |          |        |  
              |        |          |               |          |        |  
              u        b          v               n          w        x
        """
        # Create the next top environment
        top_env = []
        # First absorb left boundary mpo
        res = einsum('zay,ura->zuyr',prev_env[0],left1)
        # Merge correct inds
        res.merge_inds([2,3])
        top_env.append(res)

        # Loop through and add bras
        for col in range(ncol):
            # Get the bra tensor
            braten = bra[col][row].copy()
            # Add identity ---------------------
            # TODO - Make sure signs are correct (will give error in symmetric case)
            D1 = braten.shape[braten.legs[0][0]]
            Z1 = braten.qn_sectors[braten.legs[0][0]]
            I1 = eye(D1,
                     Z1,
                     is_symmetric=braten.is_symmetric,
                     backend=braten.backend)
            if len(braten.legs[0]) > 1:
                for legind in range(1,len(braten.legs[0])):
                    Dli = braten.shape[braten.legs[0][legind]]
                    Zli = braten.qn_sectors[braten.legs[0][legind]]
                    Ii = eye(Dli,
                             Zli,
                             is_symmetric=braten.is_symmetric,
                             backend=braten.backend)
                    I1 = einsum('ij,IJ->iIjJ',I1,Ii)
                    I1.merge_inds([0,1])
                    I1.merge_inds([1,2])
            # Contract with previous environment
            res = einsum('ybx,Rr->yRbxr',prev_env[2*col+1],I1)
            # Merge correct indices
            res.merge_inds([0,1])
            res.merge_inds([2,3])
            # Add to top_env
            top_env.append(res)

            # Add bra --------------------------
            envten = prev_env[2*col+2].copy()
            # Unmerge physical index
            if thermal:
                envten.unmerge_ind(1)
                envten.merge_inds([1,2])
            else:
                envten.unmerge_ind(1)
            # Contract with bra
            res = einsum('xlcw,rvlsc->xrvws',envten,braten)
            # Merge correct inds
            res.merge_inds([0,1])
            res.merge_inds([2,3])
            # Add to top_env
            top_env.append(res)

        # Last, absorb right boundary mpo
        res = einsum('ufz,xtf->utxz',prev_env[2*ncol+1],right1)
        # Merge needed inds
        res.merge_inds([0,1])
        # Add to top_env
        top_env.append(res)

        # Put result into an MPS ------------------
        top_env = MPS(top_env)

        # Reduce bond dimension
        if truncate:
            mpiprint(5,'Truncating Boundary MPS')
            if DEBUG:
                mpiprint(6,'Computing initial bmpo norm')
                norm0 = top_env.norm()
            top_env = top_env.apply_svd(chi)
            if DEBUG:
                mpiprint(6,'Computing resulting bmpo norm')
                norm1 = top_env.norm()
                mpiprint(0,'Norm Difference for chi={}: {}'.format(chi,abs(norm0-norm1)/abs(norm0)))

    return top_env

def calc_top_envs_gen(bra,left_bmpo,right_bmpo,ket=None,chi=10):
    """
    """
    # Figure out height of peps column
    Ny = len(bra[0])

    # Copy bra if needed
    copy_ket = False
    if ket is None: copy_ket = True
    elif hasattr(ket,'__len__'):
        if ket[0] is None: copy_ket = True
    if copy_ket:
        ket = [None]*len(bra)
        for i in range(len(bra)):
            ketcol = [None]*len(bra[i])
            for j in range(len(bra[i])):
                ketcol[j] = bra[i][j].copy()
                # TODO - Conjugate this ket col?
            ket[i] = ketcol

    # Compute top environment
    top_env = [None]*Ny
    for row in reversed(range(Ny)):
        # Figure out previous environment MPO
        if row == Ny-1: prev_env = None
        else: prev_env = top_env[row+1]
        # Compute next environment MPO
        top_env[row] = update_top_env_gen(row,
                                          bra,
                                          ket,
                                          left_bmpo[2*row],
                                          left_bmpo[2*row+1],
                                          right_bmpo[2*row],
                                          right_bmpo[2*row+1],
                                          prev_env,
                                          chi=chi)
    return top_env

def update_bot_env_gen(row,bra,ket,left1,left2,right1,right2,prev_env,chi=10,truncate=True):
    """
    Doing the following contraction:

              s        t          l               v          n        x  
              |        |          |               |          |        |  
              |        |          |               |          |        |  
           +----+   +----+        |            +----+        |     +----+
           | l2 |-p-| k1 |----q---^----  ...  -| k2 |---r----^-----| r2 |
           +----+   +----+        |            +----+        |     +----+  
              |        |  \       |               |  \       |        |   
              |        |   \      |               |   \      |        |   
              j        |    k     |               |    m     |        o    
              |        |     \    |               |     \    |        |  
              |        |      \   |               |      \   |        |   
           +----+      |       +----+             |       +----+   +----+
           | l1 |---g--^-------| b1 |-h  ...  ----^-------| b2 |-i-| r1 |
           +----+      |       +----+             |       +----+   +----+
              |        |          |               |          |        |  
              a        b          c               d          e        f  
              |        |          |               |          |        |  
           +----+   +----+     +----+           +----+    +----+   +----+   
        z--| p1 |-y-| p2 |--x--| p3 |-w  ...  --| p4 |--v-| p5 |-y-| p6 |--t
           +----+   +----+     +----+           +----+    +----+   +----+   

    """
    # Figure out number of columns
    ncol = len(bra)
    
    # Determine if it is a thermal state
    thermal = len(bra[0][row].legs[2]) == 2

    # Create the new top environment
    if prev_env is None:
        # Create the first top environment
        bot_env = []

        # First site is the current left bound_mpo
        res = einsum('agj,jps->asgp',left1,left2)
        # Merge correct inds
        res.merge_inds([2,3])
        # Add to bot env
        bot_env.append(res)

        for col in range(ncol):
            # Copy the needed tensors
            ketten = ket[col][row].copy()
            braten = bra[col][row].copy()

            # Add ket -----------------------------------
            # Remove bottom index
            ketten = ketten.remove_empty_ind(1)
            # Create needed identity
            # TODO - Make sure signs are correct (will give error in symmetric case)
            Dl = braten.shape[braten.legs[0][0]]
            Zl = braten.qn_sectors[braten.legs[0][0]]
            I = eye(Dl,
                    Zl,
                    is_symmetric=braten.is_symmetric,
                    backend=braten.backend)
            if len(braten.legs[0]) > 1:
                for legind in range(1,len(braten.legs[0])):
                    Dli = braten.shape[braten.legs[0][legind]]
                    Zli = braten.qn_sectors[braten.legs[0][legind]]
                    Ii = eye(Dli,
                             Zli,
                             is_symmetric=braten.is_symmetric,
                             backend=braten.backend)
                    I = einsum('ij,IJ->iIjJ',I,Ii)
                    I.merge_inds([0,1])
                    I.merge_inds([1,2])
            # Contract identity with the ket
            res = einsum('pkqt,Gg->Gptgkq',ketten,I)
            # Merge correct inds
            res.merge_inds([0,1])
            res.merge_inds([2,3,4])
            # Add to bot_env
            bot_env.append(res)

            # Add bra ----------------------------------
            # Remove bottom index
            braten = braten.remove_empty_ind(1)
            # Create correect identity
            # TODO - Make sure signs are correct (will give error in symmetric case)
            Dl = ketten.shape[ketten.legs[2][0]]
            Zl = ketten.qn_sectors[ketten.legs[2][0]]
            I = eye(Dl,
                    Zl,
                    is_symmetric=ketten.is_symmetric,
                    backend=ketten.backend)
            if len(ketten.legs[2]) > 1:
                for legind in range(1,len(ketten.legs[2])):
                    Dli = ketten.shape[ketten.legs[2][legind]]
                    Zli = ketten.qn_sectors[ketten.legs[2][legind]]
                    Ii = eye(Dli,
                             Zli,
                             is_symmetric=braten.is_symmetric,
                             backend=braten.backend)
                    I = einsum('ij,IJ->iIjJ',I,Ii)
                    I.merge_inds([0,1])
                    I.merge_inds([1,2])
            # Contract identity with the bra
            res = einsum('gkhl,Qq->gkQlhq',braten,I)
            # Merge correct inds
            res.merge_inds([0,1,2])
            res.merge_inds([2,3])
            # Add to bot_env
            bot_env.append(res)

        # Last site is the current right bound_mpo
        res = einsum('fio,orx->irxf',right1,right2)
        # Merge correct inds
        res.merge_inds([0,1])
        # Add to bot env
        bot_env.append(res)

        # Put result into an MPS -------------------------------------------
        bot_env = MPS(bot_env)

        # Reduce bond dimension
        if truncate:
            mpiprint(5,'Truncating Boundary MPS')
            if DEBUG:
                mpiprint(6,'Computing initial bmpo norm')
                norm0 = bot_env.norm()
            bot_env = bot_env.apply_svd(chi)
            if DEBUG:
                mpiprint(6,'Computing resulting bmpo norm')
                norm1 = bot_env.norm()
                mpiprint(0,'Norm Difference for chi={}: {}'.format(chi,abs(norm0-norm1)/abs(norm0)))
    else:
        # Add the bra layer --------------------------------------------------
        """
        Doing the following contraction:

              j        bk         l               vm         n        x    
              |        ||         |               ||         |        |  
              |        ||         |               ||         |        |   
           +----+      |+------+----+             |+------+----+   +----+
           | l1 |---g--^-------| b1 |-h  ...  ----^-------| b2 |-i-| r1 |
           +----+      |       +----+             |       +----+   +----+
              |        |          |               |          |        |  
              a        b          c               d          e        f  
              |        |          |               |          |        |  
           +----+   +----+     +----+           +----+    +----+   +----+   
        z--| p1 |-y-| p2 |--x--| p3 |-w  ...  --| p4 |--v-| p5 |-u-| p6 |--t
           +----+   +----+     +----+           +----+    +----+   +----+   
        """
        # Create the next bot environment
        bot_env = []
        # First, absorb left boundary mps
        res = einsum('agj,zay->zjyg',left1,prev_env[0])
        # Merge correct inds
        res.merge_inds([2,3])
        # Add to bottom env
        bot_env.append(res)

        # Loop through to add bras
        for col in range(ncol):
            braten = bra[col][row].copy()
            # Add identity ---------------------
            # TODO - Make sure signs are correct (will give error in symmetric case)
            D1 = braten.shape[braten.legs[0][0]]
            Z1 = braten.qn_sectors[braten.legs[0][0]]
            I1 = eye(D1,
                     Z1,
                     is_symmetric=braten.is_symmetric,
                     backend=braten.backend)
            if len(braten.legs[0]) > 1:
                for legind in range(1,len(braten.legs[0])):
                    Dli = braten.shape[braten.legs[0][legind]]
                    Zli = braten.qn_sectors[braten.legs[0][legind]]
                    Ii = eye(Dli,
                             Zli,
                             is_symmetric=braten.is_symmetric,
                             backend=braten.backend)
                    I1 = einsum('ij,IJ->iIjJ',I1,Ii)
                    I1.merge_inds([0,1])
                    I1.merge_inds([1,2])
            D2 = braten.shape[braten.legs[2][0]]
            Z2 = braten.qn_sectors[braten.legs[2][0]]
            I2 = eye(D2,
                     Z2,
                     is_symmetric=braten.is_symmetric,
                     backend=braten.backend)
            if len(braten.legs[2]) > 1:
                for legind in range(1,len(braten.legs[2])):
                    Dli = braten.shape[braten.legs[2][legind]]
                    Zli = braten.qn_sectors[braten.legs[2][legind]]
                    Ii = eye(Dli,
                             Zli,
                             is_symmetric=braten.is_symmetric,
                             backend=braten.backend)
                    I2 = einsum('ij,IJ->iIjJ',I2,Ii)
                    I2.merge_inds([0,1])
                    I2.merge_inds([1,2])
            # Contract with previous environment
            res = einsum('ybx,Gg->yGbxg',prev_env[2*col+1],I1)
            res = einsum('yGbxg,Kk->yGbKxgk',res,I2)
            # Merge correct indices
            res.merge_inds([0,1])
            res.merge_inds([1,2])
            res.merge_inds([2,3,4])
            # Add to bot_env
            bot_env.append(res)

            # Add ket --------------------------
            # Contract with previous bot_env
            res = einsum('gckhl,xcw->xgklwh',braten,prev_env[2*col+2])
            # Merge correct indices
            res.merge_inds([0,1,2])
            res.merge_inds([2,3])
            # Add to bot_env
            bot_env.append(res)

        # Last, absorb right boundary mpo
        res = einsum('fix,uft->uixt',right1,prev_env[2*ncol+1])
        # Merge needed inds
        res.merge_inds([0,1])
        # Add to bot_env
        bot_env.append(res)

        # Put result into an MPS ------------------
        bot_env = MPS(bot_env)

        # Reduce bond dimension
        if truncate:
            mpiprint(5,'Truncating Boundary MPS')
            if DEBUG:
                mpiprint(6,'Computing initial bmpo norm')
                norm0 = bot_env.norm()
            bot_env = bot_env.apply_svd(chi)
            if DEBUG:
                mpiprint(6,'Computing resulting bmpo norm')
                norm1 = bot_env.norm()
                mpiprint(0,'Norm Difference for chi={}: {}'.format(chi,abs(norm0-norm1)/abs(norm0)))

        # Update prev_env
        prev_env = bot_env

        # Add the bra layer --------------------------------------------------
        """
        Doing the following contraction:
              s        t                          v                   x  
              |        |          |               |          |        |  
              |        |          |               |          |        |  
           +----+   +----+        |             +----+       |     +----+
           | l2 |-p-| k1 |----q---^----  ...  --| k2 |---r---^-----| r2 |
           +----+   +----+        |             +----+       |     +----+  
              |       ||          |               ||         |        |  
              a       bk          c               dm         e        f  
              |       ||          |               ||         |        |  
           +----+   +----+     +----+           +----+    +----+   +----+   
        z--| p1 |-y-| p2 |--x--| p3 |-w  ...  --| p4 |--v-| p5 |-y-| p6 |--t
           +----+   +----+     +----+           +----+    +----+   +----+   
        """
        # Create the next bottom environment
        bot_env = []
        # First, absorb left boundary mpo
        res = einsum('zay,aps->zsyp',prev_env[0],left2)
        # Merge correct inds
        res.merge_inds([2,3])
        # Add to bot_env
        bot_env.append(res)

        # Loop through and add ket tensors
        for col in range(ncol):
            # Get the ket tensor
            ketten = ket[col][row].copy()
            # Add ket --------------------------
            envten = prev_env[2*col+1].copy()
            # Unmerge physical index
            if thermal:
                envten.unmerge_ind(1)
                envten.merge_inds([2,3])
            else:
                envten.unmerge_ind(1)
            # Contract with ket
            res = einsum('ybkx,pbkqt->yptxq',envten,ketten)
            # Merge correct indices
            res.merge_inds([0,1])
            res.merge_inds([2,3])
            # Add to bot_env
            bot_env.append(res)

            # Add identity ---------------------
            # TODO - Make sure signs are correct (will give error in symmetric case)
            D1 = ketten.shape[ketten.legs[3][0]]
            Z1 = ketten.qn_sectors[ketten.legs[3][0]]
            I1 = eye(D1,
                     Z1,
                     is_symmetric=ketten.is_symmetric,
                     backend=ketten.backend)
            if len(ketten.legs[3]) > 1:
                for legind in range(1,len(ketten.legs[3])):
                    Dli = ketten.shape[ketten.legs[3][legind]]
                    Zli = ketten.qn_sectors[ketten.legs[3][legind]]
                    Ii = eye(Dli,
                             Zli,
                             is_symmetric=braten.is_symmetric,
                             backend=braten.backend)
                    I1 = einsum('ij,IJ->iIjJ',I1,Ii)
                    I1.merge_inds([0,1])
                    I1.merge_inds([1,2])
            # Contract with previous environment
            res = einsum('xcw,Qq->xQcwq',prev_env[2*col+2],I1)
            # Merge correct indices
            res.merge_inds([0,1])
            res.merge_inds([2,3])
            # Add to bot_env
            bot_env.append(res)

        # Last, absorb right boundary mpo
        res = einsum('yft,frx->yrxt',prev_env[2*ncol+1],right2)
        # Merge needed inds
        res.merge_inds([0,1])
        # Add to bot_env
        bot_env.append(res)

        # Put result into an MPS ------------------
        bot_env = MPS(bot_env)

        # Reduce bond dimension
        if truncate:
            mpiprint(5,'Truncating Boundary MPS')
            if DEBUG:
                mpiprint(6,'Computing initial bmpo norm')
                norm0 = bot_env.norm()
            bot_env = bot_env.apply_svd(chi)
            if DEBUG:
                mpiprint(6,'Computing resulting bmpo norm')
                norm1 = bot_env.norm()
                mpiprint(0,'Norm Difference for chi={}: {}'.format(chi,abs(norm0-norm1)/abs(norm0)))

    # return result
    return bot_env

def calc_bot_envs_gen(bra,left_bmpo,right_bmpo,ket=None,chi=10):
    """
    """
    Ny = len(bra[0])

    # Copy bra if needed
    copy_ket = False
    if ket is None: copy_ket = True
    elif hasattr(ket,'__len__'):
        if ket[0] is None: copy_ket = True
    if copy_ket:
        ket = [None]*len(bra)
        for i in range(len(bra)):
            ketcol = [None]*len(bra[i])
            for j in range(len(bra[i])):
                ketcol[j] = bra[i][j].copy()
                # TODO - Conjugate this ket col?
            ket[i] = ketcol

    # Compute the bottom environment
    bot_env = [None]*Ny
    for row in range(Ny):
        if row == 0: prev_env = None
        else: prev_env = bot_env[row-1]
        bot_env[row] = update_bot_env_gen(row,
                                          bra,
                                          ket,
                                          left_bmpo[2*row],
                                          left_bmpo[2*row+1],
                                          right_bmpo[2*row],
                                          right_bmpo[2*row+1],
                                          prev_env,
                                          chi=chi)
    return bot_env

def update_top_env2(row,bra,ket,left1,left2,right1,right2,prev_env,chi=10,truncate=True,contracted_env=False):
    """
    Doing the following contraction:

       +-----------------------------------------------+
       |                prev_env                       |
       +-----------------------------------------------+
       |        |          |       |          |        |  
       a        b          c       d          e        f  
       |        |          |       |          |        |  
    +----+   +----+        |    +----+        |     +----+
    | l2 |-g-| k1 |-----h--^----| k2 |-----i--^-----| r2 |
    +----+   +----+        |    +----+        |     +----+  
       |        |  \       |       |  \       |        |   
       |        |   \      |       |   \      |        |   
       j        |    l     |       |    o     |        q    
       |        |     \    |       |     \    |        |  
       |        |      \   |       |      \   |        |   
    +----+      |       +----+     |       +----+   +----+
    | l1 |---r--^-------| b1 |-----^-s-----| b2 |-t-| r1 |
    +----+      |       +----+     |       +----+   +----+
       |        |          |       |          |        |  
       |        |          |       |          |        |  
       u        k          v       n          w        x

    """
    if not contracted_env:
        top_env = update_top_env_gen(row,
                                     bra,
                                     ket,
                                     left1,
                                     left2,
                                     right1,
                                     right2,
                                     prev_env,
                                     chi=chi,
                                     truncate=truncate)
    else:
        bra1 = bra[0][row]
        bra2 = bra[1][row]
        ket1 = ket[0][row]
        ket2 = ket[1][row]
        if prev_env is None:
            # Create first top env
            tmp = einsum('jga,gklhb->abjklh',left2,ket1).remove_empty_ind(0).remove_empty_ind(0)
            tmp = einsum('jklh,hnoid->djklnoi',tmp,ket2).remove_empty_ind(0)
            tmp = einsum('jklnoi,qif->fjklnoq',tmp,right2).remove_empty_ind(0)
            tmp = einsum('jklnoq,urj->urklnoq',tmp,left1)
            tmp = einsum('urklnoq,rvlsc->cukvsnoq',tmp,bra1).remove_empty_ind(0)
            tmp = einsum('ukvsnoq,swote->eukvnwtq',tmp,bra2).remove_empty_ind(0)
            top_env = einsum('ukvnwtq,xtq->ukvnwx',tmp,right1)
        else:
            tmp = einsum('jga,abcdef->jgbcdef',left2,prev_env)
            tmp = einsum('jgbcdef,gklhb->jklhcdef',tmp,ket1)
            tmp = einsum('jklhcdef,hnoid->jklcnoief',tmp,ket2)
            tmp = einsum('jklcnoief,qif->jklcnoeq',tmp,right2)
            tmp = einsum('jklcnoeq,urj->urklcnoeq',tmp,left1)
            tmp = einsum('urklcnoeq,rvlsc->ukvnsoeq',tmp,bra1)
            tmp = einsum('ukvnsoeq,swote->ukvnwtq',tmp,bra2)
            top_env = einsum('ukvnwtq,xtq->ukvnwx',tmp,right1)
    return top_env

def calc_top_envs2(bra,left_bmpo,right_bmpo,ket=None,chi=10,truncate=True,contracted_env=False):
    """
    """
    # Figure out height of peps column
    Ny = len(bra[0])

    # Copy bra if needed
    copy_ket = False
    if ket is None: copy_ket = True
    elif hasattr(ket,'__len__'):
        if ket[0] is None: copy_ket = True
    if copy_ket:
        ket = [None]*len(bra)
        for i in range(len(bra)):
            ketcol = [None]*len(bra[i])
            for j in range(len(bra[i])):
                ketcol[j] = bra[i][j].copy()
                # TODO - Conjugate this ket col?
            ket[i] = ketcol

    # Compute the bottom environment
    top_env = [None]*Ny
    for row in reversed(range(Ny)):
        if row == Ny-1: prev_env = None
        else: prev_env = top_env[row+1]
        top_env[row] = update_top_env2(row,
                                       bra,
                                       ket,
                                       left_bmpo[2*row],
                                       left_bmpo[2*row+1],
                                       right_bmpo[2*row],
                                       right_bmpo[2*row+1],
                                       prev_env,
                                       chi=chi,
                                       truncate=truncate,
                                       contracted_env=contracted_env)
    return top_env

def update_bot_env2(row,bra,ket,left1,left2,right1,right2,prev_env,chi=10,truncate=True,contracted_env=False):
    """
    Doing the following contraction:

       s        t          l       v          n        x  
       |        |          |       |          |        |  
       |        |          |       |          |        |  
    +----+   +----+        |    +----+        |     +----+
    | l2 |-p-| k1 |----q---^----| k2 |---r----^-----| r2 |
    +----+   +----+        |    +----+        |     +----+  
       |        |  \       |       |  \       |        |   
       |        |   \      |       |   \      |        |   
       j        |    k     |       |    m     |        o    
       |        |     \    |       |     \    |        |  
       |        |      \   |       |      \   |        |   
    +----+      |       +----+     |       +----+   +----+
    | l1 |---g--^-------| b1 |--h--^-------| b2 |-i-| r1 |
    +----+      |       +----+     |       +----+   +----+
       |        |          |       |          |        |  
       a        b          c       d          e        f  
       |        |          |       |          |        |  
       +-----------------------------------------------+
       |                prev_env                       |
       +-----------------------------------------------+

    """
    if not contracted_env:
        bot_env = update_bot_env_gen(row,
                                     bra,
                                     ket,
                                     left1,
                                     left2,
                                     right1,
                                     right2,
                                     prev_env,
                                     chi=chi,
                                     truncate=truncate)
    else:
        bra1 = bra[0][row]
        bra2 = bra[1][row]
        ket1 = ket[0][row]
        ket2 = ket[1][row]
        if prev_env is None:
            tmp = einsum('agj,gckhl->acjklh',left1,bra1).remove_empty_ind(0).remove_empty_ind(0)
            tmp = einsum('jklh,hemin->ejklmni',tmp,bra2).remove_empty_ind(0)
            tmp = einsum('jklmni,fio->fjklmno',tmp,right1).remove_empty_ind(0)
            tmp = einsum('jklmno,jps->spklmno',tmp,left2)
            tmp = einsum('spklmno,pbkqt->bstqlmno',tmp,ket1).remove_empty_ind(0)
            tmp = einsum('stqlmno,qdmrv->dstlvrno',tmp,ket2).remove_empty_ind(0)
            bot_env = einsum('stlvrno,orx->stlvnx',tmp,right2)
        else:
            tmp = einsum('agj,abcdef->jgbcdef',left1,prev_env)
            tmp = einsum('jgbcdef,gckhl->jbklhdef',tmp,bra1)
            tmp = einsum('jbklhdef,hemin->jbkldmnif',tmp,bra2)
            tmp = einsum('jbkldmnif,fio->jbkldmno',tmp,right1)
            tmp = einsum('jbkldmno,jps->spbkldmno',tmp,left2)
            tmp = einsum('spbkldmno,pbkqt->stqldmno',tmp,ket1)
            tmp = einsum('stqldmno,qdmrv->stlvrno',tmp,ket2)
            bot_env = einsum('stlvrno,orx->stlvnx',tmp,right2)
    return bot_env

def calc_bot_envs2(bra,left_bmpo,right_bmpo,ket=None,chi=10,truncate=True,contracted_env=False):
    """
    """
    # Figure out height of peps column
    Ny = len(bra[0])

    # Copy bra if needed
    copy_ket = False
    if ket is None: copy_ket = True
    elif hasattr(ket,'__len__'):
        if ket[0] is None: copy_ket = True
    if copy_ket:
        ket = [None]*len(bra)
        for i in range(len(bra)):
            ketcol = [None]*len(bra[i])
            for j in range(len(bra[i])):
                ketcol[j] = bra[i][j].copy()
                # TODO - Conjugate this ket col?
            ket[i] = ketcol

    # Compute the bottom environment
    bot_env = [None]*Ny
    for row in range(Ny):
        if row == 0: prev_env = None
        else: prev_env = bot_env[row-1]
        bot_env[row] = update_bot_env2(row,
                                       bra,
                                       ket,
                                       left_bmpo[2*row],
                                       left_bmpo[2*row+1],
                                       right_bmpo[2*row],
                                       right_bmpo[2*row+1],
                                       prev_env,
                                       chi=chi,
                                       truncate=truncate,
                                       contracted_env=contracted_env)
    return bot_env

def update_top_env(bra,ket,left1,left2,right1,right2,prev_env):
    """
    Doing the following contraction:

     +-------+-------+-------+
     |       |       |       |
     O       u       |       o
     |       |       |       |
     +---l---+---r---^-------+
     |       |\      |       |
     |       | \     |       |
     N       |   p   U       n
     |       |     \ |       |
     |       |      \|       |
     +-------^---L---+---R---+
     |       |       |       |
     M       d       D       m

    """
    if prev_env is None:
        # Create first top env
        tmp = einsum('ldpru,NlO->uONdpr',ket,left2).remove_empty_ind(0).remove_empty_ind(0)
        tmp = einsum('Ndpr,nro->oNdpn',tmp,right2).remove_empty_ind(0)
        tmp = einsum('Ndpn,LDpRU->UNdLDRn',tmp,bra).remove_empty_ind(0)
        tmp = einsum('NdLDRn,MLN->MdDRn',tmp,left1)
        top_env = einsum('MdDRn,mRn->MdDm',tmp,right1)
    else:
        # Add on to top env
        tmp = einsum('ldpru,OuUo->OldprUo',ket,prev_env)
        tmp = einsum('OldprUo,NlO->NdprUo',tmp,left2)
        tmp = einsum('NdprUo,nro->NdpUn',tmp,right2)
        tmp = einsum('NdpUn,LDpRU->NdLDRn',tmp,bra)
        tmp = einsum('NdLDRn,MLN->MdDRn',tmp,left1)
        top_env = einsum('MdDRn,mRn->MdDm',tmp,right1)
    return top_env

def calc_top_envs(bra_col,left_bmpo,right_bmpo,ket_col=None):
    """
    Doing the following contraction:

     +-------+-------+-------+
     |       |       |       |
     O       U       |       o
     |       |       |       |
     +---L---+---R---^-------+
     |       |\      |       |
     |       | \     |       |
     N       D   P   u       n
     |             \ |       |
     |              \|       |
     +-------l-------+---r---+
     |               |       |
     M               d       m

    """

    # Figure out height of peps column
    Ny = len(bra_col)

    # Copy bra if needed
    if ket_col is None: 
        ket_col = [None]*len(bra_col)
        for i in range(len(ket_col)):
            ket_col[i] = bra_col[i].copy()
    # TODO - Conjugate this ket col?

    # Compute top environment
    top_env = [None]*Ny
    for row in reversed(range(Ny)):
        if row == Ny-1: prev_env = None
        else: prev_env = top_env[row+1]
        top_env[row] = update_top_env(bra_col[row],
                                      ket_col[row],
                                      left_bmpo[2*row],
                                      left_bmpo[2*row+1],
                                      right_bmpo[2*row],
                                      right_bmpo[2*row+1],
                                      prev_env)
    return top_env

def update_bot_env(bra,ket,left1,left2,right1,right2,prev_env):
    """
    Doing the following contraction:

     O       u       |       o
     |       |       |       |
     |       |       |       |
     +---l---+---r---^-------+
     |       |\      |       |
     |       | \     |       |
     N       d   P   U       n
     |       |    \  |       |
     |       |     \ |       |
     +-------^---L---+---R---+
     |       |       |       |
     |       |       |       |
     M       |       D       m
     |       |       |       |
     +-------+-------+-------+

    """
    if prev_env is None:
        tmp = einsum('LDPRU,MLN->DMNPUR',bra,left1).remove_empty_ind(0).remove_empty_ind(0)
        tmp = einsum('NPUR,mRn->mNPUn',tmp,right1).remove_empty_ind(0)
        tmp = einsum('NPUn,ldPru->dNlurUn',tmp,ket).remove_empty_ind(0)
        tmp = einsum('NlurUn,NlO->OurUn',tmp,left2)
        bot_env = einsum('OurUn,nro->OuUo',tmp,right2)
    else:
        tmp = einsum('LDPRU,MdDm->MdLPURm',bra,prev_env)
        tmp = einsum('MdLPURm,MLN->NdPURm',tmp,left1)
        tmp = einsum('NdPURm,mRn->NdPUn',tmp,right1)
        tmp = einsum('NdPUn,ldPru->NlurUn',tmp,ket)
        tmp = einsum('NlurUn,NlO->OurUn',tmp,left2)
        bot_env = einsum('OurUn,nro->OuUo',tmp,right2)
    return bot_env

def calc_bot_envs(bra_col,left_bmpo,right_bmpo,ket_col=None):
    """
    Doing the following contraction:

     O       u       |       o
     |       |       |       |
     |       |       |       |
     +---l---+---r---^-------+
     |       |\      |       |
     |       | \     |       |
     N       d   P   U       n
     |       |     \ |       |
     |       |      \|       |
     +-------^---L---+---R---+
     |       |       |       |
     |       |       |       |
     M       |       D       m
     |       |       |       |
     +-------+-------+-------+

    """

    # Figure out height of peps column
    Ny = len(bra_col)

    # Copy bra if needed
    if ket_col is None: 
        ket_col = [None]*len(bra_col)
        for i in range(len(ket_col)):
            ket_col[i] = bra_col[i].copy()

    # Compute the bottom environment
    bot_env = [None]*Ny
    for row in range(Ny):
        if row == 0: prev_env = None
        else: prev_env = bot_env[row-1]
        bot_env[row] = update_bot_env(bra_col[row],
                                      ket_col[row],
                                      left_bmpo[2*row],
                                      left_bmpo[2*row+1],
                                      right_bmpo[2*row],
                                      right_bmpo[2*row+1],
                                      prev_env)
    return bot_env

def reduce_tensors(peps1,peps2):
    """
    Reduce the two peps tensors, i.e. pull off physical index
    """

    if DEBUG:
        # Figure out combined tensor (for check)
        original = einsum('LDPRU,lUpru->lLDPRpru',peps1,peps2)

    # Reduce bottom tensor
    peps1 = peps1.transpose([0,1,3,2,4])
    output = peps1.svd(3,return_ent=False,return_wgt=False)
    (ub,sb,vb) = peps1.svd(3,return_ent=False,return_wgt=False)
    phys_b = einsum('ab,bPU->aPU',sb,vb)

    # Reduce top tensor
    peps2 = peps2.transpose([1,2,0,3,4])
    (ut,st,vt) = peps2.svd(2,return_ent=False,return_wgt=False)
    phys_t = einsum('DPa,ab->DPb',ut,st)
    vt = vt.transpose([1,0,2,3])

    if DEBUG:
        # Check to make sure initial and reduced peps tensors are identical
        final = einsum('LDRa,aPb->LDRPb',ub,phys_b)
        final = einsum('LDRPb,bpc->LDRPpc',final,phys_t)
        final = einsum('LDRPpc,lcru->lLDPRpru',final,vt)
        mpiprint(0,'Reduced Difference = {}'.format((original-final).abs().sum()))

    # Return result
    return ub,phys_b,phys_t,vt

def pos_sqrt_vec(vec):
    """
    """
    for i in range(vec.shape[0]):
        if vec[i] > 0.:
            vec[i] = vec[i]**(1./2.)
        else:
            vec[i] = 0.
    return vec

def make_N_positive(N,hermitian=True,positive=True):
    """
    """

    # Get a hermitian approximation of the environment
    if hermitian:
        N1 = N.copy()
        N1 = N1.transpose([0,2,1,3])
        N = N.transpose([1,3,0,2])
        N = (N+N1)/2.
        N1 = N.copy()
        N = einsum('UDab,abud->UuDd',N,N1)

    # Get a positive approximation of the environment
    if positive:
        try:
            if N.sym is None:
                N = N.transpose([0,2,1,3])
                n1 = np.prod([N.ten.shape[i] for i in N.legs[0]])
                n2 = np.prod([N.ten.shape[i] for i in N.legs[1]])
                n3 = np.prod([N.ten.shape[i] for i in N.legs[2]])
                n4 = np.prod([N.ten.shape[i] for i in N.legs[3]])
                Nmat = N.backend.reshape(N.ten,(n1*n2,n3*n4))
                u,v = N.backend.eigh(Nmat)
                u = pos_sqrt_vec(u)
                Nmat = N.backend.einsum('ij,j,kj->ik',v,u,v)
                N.ten = Nmat.reshape(N.shape)
                N = N.transpose([0,2,1,3])
            else:
                N = N.copy().transpose([0,2,1,3])
                # Make this a sparse tensor
                Nmat = N.ten.make_sparse()
                (N1,N2,N3,N4,n1,n2,n3,n4) = Nmat.shape
                Nmat = Nmat.transpose([0,4,1,5,2,6,3,7])
                Nmat = Nmat.reshape((N1*n1*N2*n2,N3*n3*N4*n4))
                u,v = N.backend.eigh(Nmat)
                u = pos_sqrt_vec(u)
                Nmat = N.backend.einsum('ij,j,kj->ik',v,u,v)
                Nmat = Nmat.reshape((N1,n1,N2,n2,N3,n3,N4,n4))
                Nmat = Nmat.transpose([0,2,4,6,1,3,5,7])
                # Cast back into a symtensor
                delta = N.ten.get_irrep_map()
                Nmat = N.backend.einsum('ABCDabcd,ABCD->ABCabcd',Nmat,delta)
                N.ten.array = Nmat
                # Retranspose
                N = N.transpose([0,2,1,3])
        except Exception as e:
            mpiprint(0,'Failed to make N positive, eigenvalues did not converge')

    return N

def calc_local_env(bra1,bra2,ket1,ket2,env_top,env_bot,lbmpo,rbmpo,reduced=True,hermitian=True,positive=True):
    """
    Calculate the local environment around two peps tensors

    Args:
        bra1 : peps tensor
            The peps tensor for the bottom site
        bra2 : peps tensor
            The peps tensor for the top site
        ket1 : peps tensor
            The peps tensor for the bottom site
        ket2 : peps tensor
            The peps tensor for the top site
        env_top : env tensor
            The top environment for the given sites
        env_bot : env tensor
            The bottom environment for the given sites
        lbmpo : list of left boundary mpo tensors
            The four left boundary mpo tensors surrounding
            the two peps tensors
        rbmpo : list of right boundary mpo tensors
            The four right boundary mpo tensors surrounding
            the two peps tensors

    Kwargs:
        reduced : bool
            If true, then this function returns the reduced
            environment. Currently, this is the only option
            available.
        hermitian : bool
            Approximate the environment with its nearest
            hermitian approximate
        positive : bool
            Approximate the environment with its nearest
            possible positive approximate

    """

    if reduced:
        # Get reduced tensors
        ub,phys_b,phys_t,vt = reduce_tensors(bra1,bra2)
        ubk,phys_bk,phys_tk,vtk = reduce_tensors(ket1,ket2)

        # Compute bottom half of environment
        if env_bot is None:
            tmp = einsum('CLB,LDRU->CDBUR',lbmpo[0],ub).remove_empty_ind(0).remove_empty_ind(0)
            tmp = einsum('BUR,cRb->cBUb',tmp,rbmpo[0]).remove_empty_ind(0)
            tmp = einsum('BUb,BlA->AlUb',tmp,lbmpo[1])
            tmp = einsum('AlUb,ldru->dAurUb',tmp,ubk).remove_empty_ind(0)
            envb= einsum('AurUb,bra->AuUa',tmp,rbmpo[1])
        else:
            tmp = einsum('CdDc,CLB->BLdDc',env_bot,lbmpo[0])
            tmp = einsum('BLdDc,LDRU->BdURc',tmp,ub)
            tmp = einsum('BdURc,cRb->BdUb',tmp,rbmpo[0])
            tmp = einsum('BdUb,BlA->AldUb',tmp,lbmpo[1])
            tmp = einsum('AldUb,ldru->AurUb',tmp,ubk)
            envb= einsum('AurUb,bra->AuUa',tmp,rbmpo[1])

        # Compute top half of environment
        if env_top is None:
            tmp = einsum('BlC,ldru->CuBdr',lbmpo[3],vtk).remove_empty_ind(0).remove_empty_ind(0)
            tmp = einsum('Bdr,brc->cBdb',tmp,rbmpo[3]).remove_empty_ind(0)
            tmp = einsum('Bdb,ALB->ALdb',tmp,lbmpo[2])
            tmp = einsum('ALdb,LDRU->UAdDRb',tmp,vt).remove_empty_ind(0)
            envt= einsum('AdDRb,aRb->AdDa',tmp,rbmpo[2])
        else:
            tmp = einsum('CuUc,BlC->BluUc',env_top,lbmpo[3])
            tmp = einsum('BluUc,ldru->BdrUc',tmp,vtk)
            tmp = einsum('BdrUc,brc->BdUb',tmp,rbmpo[3])
            tmp = einsum('BdUb,ALB->ALdUb',tmp,lbmpo[2])
            tmp = einsum('ALdUb,LDRU->AdDRb',tmp,vt)
            envt= einsum('AdDRb,aRb->AdDa',tmp,rbmpo[2])

        # Compute Environment
        N = einsum('AdDa,AuUa->uUdD',envt,envb)
        N = make_N_positive(N,hermitian=hermitian,positive=positive)

        return ub,phys_b,phys_t,vt,ubk,phys_bk,phys_tk,vtk,N
    else:
        raise NotImplementedError()

def calc_local_op(phys_b_bra,phys_t_bra,N,ham,
                      phys_b_ket=None,phys_t_ket=None,
                      reduced=True,normalize=True,return_norm=False):
    """
    Calculate the normalized Energy of the system
    """
    # Make some copies
    if phys_t_ket is None:
        phys_t_ket = phys_t_bra.copy().conj()
    if phys_b_ket is None:
        phys_b_ket = phys_b_bra.copy().conj()

    # Compute Energy (or op value
    if reduced:
        tmp = einsum('APU,UQB->APQB',phys_b_bra,phys_t_bra)
        tmp1= einsum('APQB,aAbB->aPQb',tmp,N)
        tmp2= einsum('apu,uqb->apqb',phys_b_ket,phys_t_ket)
        norm = einsum('apqb,apqb->',tmp1,tmp2)
        if ham is not None:
            tmp = einsum('aPQb,apqb->PQpq',tmp1,tmp2)
            if len(tmp.legs[0]) == 2:
                # Thermal state
                tmp.unmerge_ind(3)
                tmp.unmerge_ind(2)
                tmp.unmerge_ind(1)
                tmp.unmerge_ind(0)
                E = einsum('PaQbpaqb,PQpq->',tmp,ham)
                tmp.merge_inds([0,1])
                tmp.merge_inds([1,2])
                tmp.merge_inds([2,3])
                tmp.merge_inds([3,4])
            else:
                # Normal peps
                E = einsum('PQpq,PQpq->',tmp,ham)
        else:
            E = norm
        mpiprint(7,'E = {}/{} = {}'.format(E,norm,E/norm))
        if normalize:
            if return_norm:
                return E/norm,norm
            else:
                return E/norm
        else:
            if return_norm:
                return E,norm
            else:
                return E
    else:
        raise NotImplementedError()

def calc_N(row,bra_col,left_bmpo,right_bmpo,top_envs,bot_envs,hermitian=True,positive=True,ket_col=None):
    """
    Calculate the environment tensor
    """
    # Copy bra if needed
    _ket_col = ket_col
    if ket_col is None: 
        ket_col = [None]*len(bra_col)
        for i in range(len(ket_col)):
            ket_col[i] = bra_col[i].copy()

    # Compute Local Environment (N)
    if row == 0:
        if len(bra_col) == 2:
            # Only two sites in column, use identity at both ends
            res = calc_local_env(bra_col[row],
                                 bra_col[row+1],
                                 ket_col[row],
                                 ket_col[row+1],
                                 None,
                                 None,
                                 left_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                 right_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                 hermitian=hermitian,
                                 positive=positive)
        else:
            # Identity only on bottom
            res = calc_local_env(bra_col[row],
                                 bra_col[row+1],
                                 ket_col[row],
                                 ket_col[row+1],
                                 top_envs[row+2],
                                 None,
                                 left_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                 right_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                 hermitian=hermitian,
                                 positive=positive)
    elif row == len(bra_col)-2:
        # Identity needed on top
        res = calc_local_env(bra_col[row],
                             bra_col[row+1],
                             ket_col[row],
                             ket_col[row+1],
                             None,
                             bot_envs[row-1],
                             left_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                             right_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                             hermitian=hermitian,
                             positive=positive)
    else:
        # Get the local environment tensor (no identity needed)
        res = calc_local_env(bra_col[row],
                             bra_col[row+1],
                             ket_col[row],
                             ket_col[row+1],
                             top_envs[row+2],
                             bot_envs[row-1],
                             left_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                             right_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                             hermitian=hermitian,
                             positive=positive)
    return res

def calc_local_nn_op_lb(mpo,bra,ket,top,bot,left,right,normalize=True,contracted_env=False,chi=10):
    """
    Calculate the value of an operator as an mpo acting on the left
    and bottom bonds of a 2x2 peps grid
    """
    # Check if it is a thermal state:
    thermal = len(bra[0][1].legs[2]) == 2
    # Absorb MPO into bra
    Hbra = [[None,None],[None,None]]
    if thermal:
        bra[0][1].unmerge_ind(2)
        Hbra[0][1] = einsum('ldparu,pPx->ldxParu',bra[0][1],mpo[0]) # Top left site
        Hbra[0][1].merge_inds([1,2])
        Hbra[0][1].merge_inds([2,3])
        bra[0][1].merge_inds([2,3])
        bra[0][0].unmerge_ind(2)
        Hbra[0][0] = einsum('ldparu,xpPy->ldParyux',bra[0][0],mpo[1]) # Bottom left site
        Hbra[0][0].merge_inds([2,3])
        Hbra[0][0].merge_inds([3,4])
        Hbra[0][0].merge_inds([4,5])
        bra[0][0].merge_inds([2,3])
        bra[1][0].unmerge_ind(2)
        Hbra[1][0] = einsum('ldparu,ypP->lydParu',bra[1][0],mpo[2]) # Bottom right site
        Hbra[1][0].merge_inds([0,1])
        Hbra[1][0].merge_inds([2,3])
        Hbra[1][1] = bra[1][1].copy()
        bra[1][0].merge_inds([2,3])
    else:
        Hbra[0][1] = einsum('ldpru,pPx->ldxPru',bra[0][1],mpo[0]) # Top left site
        Hbra[0][1].merge_inds([1,2])
        Hbra[0][0] = einsum('ldpru,xpPy->ldPryux',bra[0][0],mpo[1]) # Bottom left site
        Hbra[0][0].merge_inds([3,4])
        Hbra[0][0].merge_inds([4,5])
        Hbra[1][0] = einsum('ldpru,ypP->lydPru',bra[1][0],mpo[2]) # Bottom right site
        Hbra[1][0].merge_inds([0,1])
        Hbra[1][1] = bra[1][1].copy()
    # Calculate Operator -------------------------------------
    # Compute bottom environment as a boundary mpo
    Hbot = update_bot_env2(0,
                           Hbra,
                           ket,
                           left[0],
                           left[1],
                           right[0],
                           right[1],
                           bot,
                           truncate=True,
                           chi=chi,
                           contracted_env=contracted_env)
    # Compute top environment as a boundary mpo
    Htop = update_top_env2(1,
                           Hbra,
                           ket,
                           left[2],
                           left[3],
                           right[2],
                           right[3],
                           top,
                           truncate=True,
                           chi=chi,
                           contracted_env=contracted_env)
    # Contract top and bottom boundary mpos to get result
    if contracted_env:
        E = einsum('lkbKBr,lkbKBr->',Hbot,Htop)
    else:
        E = Hbot.contract(Htop)
    # Calculate Norm -------------------------------------
    if normalize:
        # Compute bottom environment as a boundary mpo
        Nbot = update_bot_env2(0,
                               bra,
                               ket,
                               left[0],
                               left[1],
                               right[0],
                               right[1],
                               bot,
                               truncate=True,
                               chi=chi,
                               contracted_env=contracted_env)
        # Compute top environment as a boundary mpo
        Ntop = update_top_env2(1,
                               bra,
                               ket,
                               left[2],
                               left[3],
                               right[2],
                               right[3],
                               top,
                               truncate=True,
                               chi=chi,
                               contracted_env=contracted_env)
        # Contract top and bottom boundary mpos to get result
        if contracted_env:
            norm = einsum('lkbKBr,lkbKBr->',Nbot,Ntop)
        else:
            norm = Nbot.contract(Ntop)
        E /= norm
    # Return result
    return E

def calc_local_nn_op_ru(mpo,bra,ket,top,bot,left,right,normalize=True,contracted_env=False,chi=10):
    """
    Calculate the value of an operator as an mpo acting on the right
    and top bonds of a 2x2 peps grid
    """
    # Check if it is a thermal state:
    thermal = len(bra[0][1].legs[2]) == 2
    # Absorb MPO into bra
    Hbra = [[None,None],[None,None]]
    if thermal:
        bra[0][1].unmerge_ind(2)
        Hbra[0][1] = einsum('ldparu,pPx->ldParxu',bra[0][1],mpo[0]) # Top Left Site
        Hbra[0][1].merge_inds([2,3])
        Hbra[0][1].merge_inds([3,4])
        bra[0][1].merge_inds([2,3])
        bra[1][1].unmerge_ind(2)
        Hbra[1][1] = einsum('ldparu,xpPy->lxdyParu',bra[1][1],mpo[1]) # Top Right Site
        Hbra[1][1].merge_inds([0,1])
        Hbra[1][1].merge_inds([1,2])
        Hbra[1][1].merge_inds([2,3])
        bra[1][1].merge_inds([2,3])
        bra[1][0].unmerge_ind(2)
        Hbra[1][0] = einsum('ldparu,ypP->ldParuy',bra[1][0],mpo[2]) # Bottom right site
        Hbra[1][0].merge_inds([2,3])
        Hbra[1][0].merge_inds([4,5])
        bra[1][0].merge_inds([2,3])
        Hbra[0][0] = bra[0][0].copy()
    else:
        Hbra[0][1] = einsum('ldpru,pPx->ldPrxu',bra[0][1],mpo[0]) # Top Left Site
        Hbra[0][1].merge_inds([3,4])
        Hbra[1][1] = einsum('ldpru,xpPy->lxdyPru',bra[1][1],mpo[1]) # Top Right Site
        Hbra[1][1].merge_inds([0,1])
        Hbra[1][1].merge_inds([1,2])
        Hbra[1][0] = einsum('ldpru,ypP->ldPruy',bra[1][0],mpo[2]) # Bottom right site
        Hbra[1][0].merge_inds([4,5])
        Hbra[0][0] = bra[0][0].copy()
    # Calculate Operator -------------------------------------
    # Compute bottom environment as a boundary mpo
    Hbot = update_bot_env2(0,
                           Hbra,
                           ket,
                           left[0],
                           left[1],
                           right[0],
                           right[1],
                           bot,
                           truncate=True,
                           chi=chi,
                           contracted_env=contracted_env)
    # Compute top environment as a boundary mpo
    Htop = update_top_env2(1,
                           Hbra,
                           ket,
                           left[2],
                           left[3],
                           right[2],
                           right[3],
                           top,
                           truncate=True,
                           chi=chi,
                           contracted_env=contracted_env)
    # Contract top and bottom boundary mpos to get result
    if contracted_env:
        E = einsum('lkbKBr,lkbKBr->',Hbot,Htop)
    else:
        E = Hbot.contract(Htop)
    # Calculate Norm -------------------------------------
    if normalize:
        # Compute bottom environment as a boundary mpo
        Nbot = update_bot_env2(0,
                               bra,
                               ket,
                               left[0],
                               left[1],
                               right[0],
                               right[1],
                               bot,
                               truncate=True,
                               chi=chi,
                               contracted_env=contracted_env)
        # Compute top environment as a boundary mpo
        Ntop = update_top_env2(1,
                               bra,
                               ket,
                               left[2],
                               left[3],
                               right[2],
                               right[3],
                               top,
                               truncate=True,
                               chi=chi,
                               contracted_env=contracted_env)
        # Contract top and bottom boundary mpos to get result
        if contracted_env:
            norm = einsum('lkbKBr,lkbKBr->',Nbot,Ntop)
        else:
            norm = Nbot.contract(Ntop)
        E /= norm
    # Return result
    return E

def calc_local_nn_op(row,bra,ops_col,left_bmpo,right_bmpo,bot_envs,top_envs,ket=None,normalize=True,contracted_env=False,chi=10):
    """
    Calculate the value of an operator on a 2x2 square
    
    Args:
        row: int
            The row of the ops_col to be evaluated
        bra: list of list of ndarrays
            The needed columns of the peps
        left_bmpo:
            The boundary mpo to the left of the two peps columns
        right_bmpo:
            The boundary mpo to the right of the two peps columns
        bot_envs:
            The boundary mpo version of the bottom environment
        top_envs:
            The boundary mpo version of the top environment
        ops_col: list of list of ndarrays
            The operators acting on next nearest neighboring sites
            within the two columns
          
    Kwargs:
        normalize: bool
            Whether to normalize the operator evaluations
        ket: List of list of ndarrays
            The needed columns of the ket
        contracted_env: bool
            Whether to contract the upper and lower environment
            or leave it as a boundary mps
        chi: int
            Max bond dimension for the boundary mps on the top
            and bottom

    Returns:
        E: float
            The operator value for the given 2x2 plaquette
    """

    # Copy bra if needed
    copy_ket = False
    if ket is None: copy_ket = True
    elif hasattr(ket,'__len__'):
        if ket[0] is None: copy_ket = True
    if copy_ket:
        ket = [None]*len(bra)
        for i in range(len(bra)):
            ketcol = [None]*len(bra[i])
            for j in range(len(bra[i])):
                ketcol[j] = bra[i][j].copy()
                # TODO - Conjugate this ket col?
            ket[i] = ketcol

    E = 0.
    if row == 0:
        if len(bra[0]) == 2:
            # Only two sites in column, use identity at both ends
            E += calc_local_nn_op_lb(ops_col[row][0],
                                     [[bra[0][row],bra[0][row+1]],[bra[1][row],bra[1][row+1]]],
                                     [[ket[0][row],ket[0][row+1]],[ket[1][row],ket[1][row+1]]],
                                     None, # top_envs[row+2]
                                     None, # bot_envs[row-1]
                                     left_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                     right_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                     normalize=normalize,
                                     chi=chi,
                                     contracted_env=contracted_env)
            E += calc_local_nn_op_ru(ops_col[row][1],
                                     [[bra[0][row],bra[0][row+1]],[bra[1][row],bra[1][row+1]]],
                                     [[ket[0][row],ket[0][row+1]],[ket[1][row],ket[1][row+1]]],
                                     None, # top_envs[row+2]
                                     None, # bot_envs[row-1]
                                     left_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                     right_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                     normalize=normalize,
                                     chi=chi,
                                     contracted_env=contracted_env)
        else:
            E += calc_local_nn_op_lb(ops_col[row][0],
                                     [[bra[0][row],bra[0][row+1]],[bra[1][row],bra[1][row+1]]],
                                     [[ket[0][row],ket[0][row+1]],[ket[1][row],ket[1][row+1]]],
                                     top_envs[row+2],
                                     None, # bot_envs[row-1]
                                     left_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                     right_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                     normalize=normalize,
                                     chi=chi,
                                     contracted_env=contracted_env)
            E += calc_local_nn_op_ru(ops_col[row][1],
                                     [[bra[0][row],bra[0][row+1]],[bra[1][row],bra[1][row+1]]],
                                     [[ket[0][row],ket[0][row+1]],[ket[1][row],ket[1][row+1]]],
                                     top_envs[row+2],
                                     None, # bot_envs[row-1]
                                     left_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                     right_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                     normalize=normalize,
                                     chi=chi,
                                     contracted_env=contracted_env)
    elif row == len(bra[0])-2:
        # Identity needed on top
        E += calc_local_nn_op_lb(ops_col[row][0],
                                 [[bra[0][row],bra[0][row+1]],[bra[1][row],bra[1][row+1]]],
                                 [[ket[0][row],ket[0][row+1]],[ket[1][row],ket[1][row+1]]],
                                 None, # top_envs[row+2]
                                 bot_envs[row-1],
                                 left_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                 right_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                 normalize=normalize,
                                 chi=chi,
                                 contracted_env=contracted_env)
        E += calc_local_nn_op_ru(ops_col[row][1],
                                 [[bra[0][row],bra[0][row+1]],[bra[1][row],bra[1][row+1]]],
                                 [[ket[0][row],ket[0][row+1]],[ket[1][row],ket[1][row+1]]],
                                 None, # top_envs[row+2]
                                 bot_envs[row-1],
                                 left_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                 right_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                 normalize=normalize,
                                 chi=chi,
                                 contracted_env=contracted_env)
    else:
        # Get the local environment tensor (no identity needed)
        E += calc_local_nn_op_lb(ops_col[row][0],
                                 [[bra[0][row],bra[0][row+1]],[bra[1][row],bra[1][row+1]]],
                                 [[ket[0][row],ket[0][row+1]],[ket[1][row],ket[1][row+1]]],
                                 top_envs[row+2],
                                 bot_envs[row-1],
                                 left_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                 right_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                 normalize=normalize,
                                 chi=chi,
                                 contracted_env=contracted_env)
        E += calc_local_nn_op_ru(ops_col[row][1],
                                 [[bra[0][row],bra[0][row+1]],[bra[1][row],bra[1][row+1]]],
                                 [[ket[0][row],ket[0][row+1]],[ket[1][row],ket[1][row+1]]],
                                 top_envs[row+2],
                                 bot_envs[row-1],
                                 left_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                 right_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                 normalize=normalize,
                                 chi=chi,
                                 contracted_env=contracted_env)
    # Return resulting energy
    return E

def calc_single_column_nn_op(peps,left_bmpo,right_bmpo,ops_col,normalize=True,ket=None,chi=10,contracted_env=False):
    """
    Calculate contribution to an operator with next nearest (nn) neighbr interactions
    from two neighboring columns of a peps

    Args:
        peps: List of list of ndarrays
            The needed columns of the peps
        left_bmpo:
            The boundary mpo to the left of the two peps columns
        right_bmpo:
            The boundary mpo to the right of the two peps columns
        ops_col: list of list of ndarrays
            The operators acting on next nearest neighboring sites
            within the two columns

    Kwargs:
        normalize: bool
            Whether to normalize the operator evaluations
        ket: List of list of ndarrays
            The needed columns of the ket
        contracted_env: bool
            Whether to contract the upper and lower environment
            or leave it as a boundary mps

    Returns:
        E: float
            The operator value for interactions between the two columns
    """
    # Calculate top and bottom environments
    top_envs = calc_top_envs2(peps,left_bmpo,right_bmpo,ket=ket,chi=chi,contracted_env=contracted_env)
    bot_envs = calc_bot_envs2(peps,left_bmpo,right_bmpo,ket=ket,chi=chi,contracted_env=contracted_env)

    # Calculate Energy
    E = peps[0][0].backend.zeros(len(ops_col))
    for row in range(len(ops_col)):
        E[row] = calc_local_nn_op(row,
                                  peps,
                                  ops_col,
                                  left_bmpo,
                                  right_bmpo,
                                  bot_envs,
                                  top_envs,
                                  ket=ket,
                                  chi=chi,
                                  normalize=normalize,
                                  contracted_env=contracted_env)
    return E

def calc_single_column_op(peps_col,left_bmpo,right_bmpo,ops_col,normalize=True,ket_col=None):
    """
    Calculate contribution to operator from interactions within
    a single column.

    Args:
        peps_col:
            A single column of the peps
        left_bmpo:
            The boundary mpo to the left of the peps column
        right_bmpo:
            The boundary mpo to the right of the peps column
        ops:
            The operators acting on nearest neighboring sites
            within the column

    """

    # Calculate top and bottom environments
    top_envs = calc_top_envs(peps_col,left_bmpo,right_bmpo,ket_col=ket_col)
    bot_envs = calc_bot_envs(peps_col,left_bmpo,right_bmpo,ket_col=ket_col)

    # Calculate Energy
    E = peps_col[0].backend.zeros(len(ops_col))
    for row in range(len(ops_col)):
        res = calc_N(row,peps_col,left_bmpo,right_bmpo,top_envs,bot_envs,hermitian=False,positive=False,ket_col=ket_col)
        _,phys_b,phys_t,_,_,phys_bk,phys_tk,_,N = res
        E[row] = calc_local_op(phys_b,phys_t,N,ops_col[row],normalize=normalize,phys_b_ket=phys_bk,phys_t_ket=phys_tk)
    return E

def calc_all_column_op(peps,ops,chi=10,return_sum=True,normalize=True,ket=None):
    """
    Calculate contribution to operator from interactions within all columns,
    ignoring interactions between columns

    Args:
        peps : A list of lists of peps tensors
            The PEPS to be normalized
        ops :
            The operator to be contracted with the peps

    Kwargs:
        chi : int
            The maximum bond dimension for the boundary mpo
        return_sum : bool
            Whether to return the summation of all energies or
            a 2D array showing the energy contribution from each bond.
        ket : A list of lists of ket tensors
            A second peps, to use as the ket, in the operator contraction

    Returns:
        val : float
            The contribution of the column's interactions to
            the observable's expectation value
    """

    # Figure out peps size
    Nx = len(peps)
    Ny = len(peps[0])

    # Compute the boundary MPOs
    right_bmpo = calc_right_bound_mpo(peps, 0,chi=chi,return_all=True,ket=ket)
    left_bmpo  = calc_left_bound_mpo (peps,Nx,chi=chi,return_all=True,ket=ket)
    ident_bmpo = identity_mps(len(right_bmpo[0]),
                              dtype=peps[0][0].dtype,
                              sym=(peps[0][0].sym is not None),
                              backend=peps.backend)

    # Loop through all columns
    E = peps.backend.zeros((len(ops),len(ops[0])),dtype=peps[0][0].dtype)
    for col in range(Nx):
        if ket is None:
            ket_col = None
        else: ket_col = ket[col]
        if col == 0:
            E[col,:] = calc_single_column_op(peps[col],ident_bmpo,right_bmpo[col],ops[col],normalize=normalize,ket_col=ket_col)
        elif col == Nx-1:
            # Use Identity on the right side
            E[col,:] = calc_single_column_op(peps[col],left_bmpo[col-1],ident_bmpo,ops[col],normalize=normalize,ket_col=ket_col)
        else:
            E[col,:] = calc_single_column_op(peps[col],left_bmpo[col-1],right_bmpo[col],ops[col],normalize=normalize,ket_col=ket_col)
    mpiprint(8,'Energy [:,:] = \n{}'.format(E))

    if return_sum:
        return E.sum()
    else:
        return E

def calc_peps_nn_op(peps,ops,chi=10,normalize=True,ket=None,contracted_env=False):
    """
    Calculate the expectation value for a given next nearest (nn) neighbor operator

    Args:
        peps : A PEPS object
            The PEPS to be normalized
        ops :
            The operator to be contracted with the peps

    Kwargs:
        chi : int
            The maximum bond dimension for the boundary mpo
        normalize : bool
            Whether to divide the resulting operator value by the peps norm
        ket : PEPS Object
            A second peps, to use as the ket, in the operator contraction
        contracted_env: bool
            Whether to contract the upper and lower environment
            or leave it as a boundary mps

    Returns:
        val : float
            The resulting observable's expectation value
    """
    # Absorb Lambda tensors if needed
    if peps.ltensors is not None:
        peps = peps.copy()
        peps.absorb_lambdas()
    else:
        peps = peps.copy()
    if ket is not None and ket.ltensors is not None:
        ket = ket.copy()
        ket.absorb_lambdas()
    elif ket is not None:
        ket = ket.copy()

    # Figure out peps size
    Nx = len(peps)
    Ny = len(peps[0])

    # Compute the boundary MPOs
    right_bmpo = calc_right_bound_mpo(peps, 0,chi=chi,return_all=True,ket=ket)
    left_bmpo  = calc_left_bound_mpo (peps,Nx,chi=chi,return_all=True,ket=ket)
    ident_bmpo = identity_mps(len(right_bmpo[0]),
                              dtype=peps[0][0].dtype,
                              sym=(peps[0][0].sym is not None),
                              backend=peps.backend)

    # Loop through all columns
    E = peps.backend.zeros((len(ops),len(ops[0])),dtype=peps[0][0].dtype)
    for col in range(Nx-1):
        # Use None if no ket tensor
        if ket is None:
            ket1 = None
            ket2 = None
        else: 
            ket1 = ket[col]
            ket2 = ket[col+1]
        # Evaluate energy for single column
        if col == 0:
            if len(peps) == 2:
                # Use identities on both sides
                E[col,:] = calc_single_column_nn_op([peps[col],peps[col+1]],
                                                    ident_bmpo,
                                                    ident_bmpo,
                                                    ops[col],
                                                    normalize=normalize,
                                                    ket=[ket1,ket2],
                                                    chi=chi,
                                                    contracted_env=contracted_env)
            else:
                # Use identity on left side
                E[col,:] = calc_single_column_nn_op([peps[col],peps[col+1]],
                                                    ident_bmpo,
                                                    right_bmpo[col+1],
                                                    ops[col],
                                                    normalize=normalize,
                                                    ket=[ket1,ket2],
                                                    chi=chi,
                                                    contracted_env=contracted_env)
        elif col == Nx-2:
            # Use Identity on the right side
            E[col,:] = calc_single_column_nn_op([peps[col],peps[col+1]],
                                                left_bmpo[col-1],
                                                ident_bmpo,
                                                ops[col],
                                                normalize=normalize,
                                                ket=[ket1,ket2],
                                                chi=chi,
                                                contracted_env=contracted_env)
        else:
            E[col,:] = calc_single_column_nn_op([peps[col],peps[col+1]],
                                                left_bmpo[col-1],
                                                right_bmpo[col+1],
                                                ops[col],
                                                normalize=normalize,
                                                ket=[ket1,ket2],
                                                chi=chi,
                                                contracted_env=contracted_env)

    # Print out results if wanted
    mpiprint(8,'Energy [:,:] = \n{}'.format(E))

    # Return Result
    return E.sum()

def calc_peps_op(peps,ops,chi=10,return_sum=True,normalize=True,ket=None):
    """
    Calculate the expectation value for a given operator

    Args:
        peps : A PEPS object
            The PEPS to be normalized
        ops :
            The operator to be contracted with the peps

    Kwargs:
        chi : int
            The maximum bond dimension for the boundary mpo
        normalize : bool
            Whether to divide the resulting operator value by the peps norm
        return_sum : bool
            Whether to either return an array of the results, the same shape
            as ops, or a summation of all operators
        ket : PEPS Object
            A second peps, to use as the ket, in the operator contraction

    Returns:
        val : float
            The resulting observable's expectation value
    """
    # Absorb Lambda tensors if needed
    if peps.ltensors is not None:
        peps = peps.copy()
        peps.absorb_lambdas()
    else:
        peps = peps.copy()
    if ket is not None and ket.ltensors is not None:
        ket = ket.copy()
        ket.absorb_lambdas()
    elif ket is not None:
        ket = ket.copy()

    # Calculate contribution from interactions between columns
    col_energy = calc_all_column_op(peps,ops[0],chi=chi,normalize=normalize,return_sum=return_sum,ket=ket)

    # Calculate contribution from interactions between rows
    peps.rotate(clockwise=True)
    if ket is not None: 
        ket.rotate(clockwise=True)
    row_energy = calc_all_column_op(peps,ops[1],chi=chi,normalize=normalize,return_sum=return_sum,ket=ket)
    peps.rotate(clockwise=False)
    if ket is not None: 
        ket.rotate(clockwise=False)

    # Return Result
    if return_sum:
        return col_energy.sum()+row_energy.sum()
    else:
        return col_energy,row_energy

def increase_peps_mbd_lambda(peps,Dnew,noise=0.01):
    """
    Increase the bond dimension of lambda tensors in a
    canonical peps

    Args:
        peps : PEPS Object
            The peps for which we are increasing the bond dimension
        Dnew : int
            The new bond dimension

    Kwargs:
        noise : float
            The maximum magnitude of random noise to be incorporated
            in increasing the bond dimension

    Returns:
        peps : PEPS Object
            The peps with the bond dimension increased
    """
    if peps.ltensors is None:
        # Do nothing if there are no lambda tensors
        return peps
    else:
        # Figure out peps size
        Nx = len(peps.ltensors[0])
        Ny = len(peps.ltensors[0][0])
        Dold = peps.ltensors[0][0][0].shape[0]

        # Get unitary tensor for insertion
        identity = zeros((Dnew,Dold),dtype=peps.ltensors[0][0][0].dtype)
        identity[:Dold,:] = eye(Dold,dtype=peps.ltensors[0][0][0].dtype)
        mat = identity + noise*rand((Dnew,Dold),dtype=peps.ltensors[0][0][0].dtype)
        mat = svd(mat)[0]

        # Loop through all possible tensors and increase their sizes
        for ind in range(len(peps.ltensors)):
            for x in range(len(peps.ltensors[ind])):
                for y in range(len(peps.ltensors[ind][x])):
                    peps.ltensors[ind][x][y] = einsum('Ll,l->L',mat,peps.ltensors[ind][x][y])

        # Return result
        return peps

def increase_zn_peps_mbd(peps,Dnew,noise=1e-10):
    raise NotImplementedError()

def increase_peps_mbd(peps,Dnew,noise=1e-10,chi=None,normalize=True):
    """
    Increase the bond dimension of a peps

    Args:
        peps : 2D Array
            The peps tensors in a list of lists
        Dnew : int
            The new bond dimension

    Kwargs:
        noise : float
            The maximum magnitude of random noise to be incorporated
            in increasing the bond dimension

    Returns:
        peps : 2D Array
            The new peps tensors with increased bond dimensions
    """
    # Separate routine if using Zn Symmetry
    if peps.Zn is not None:
        return increase_zn_peps_mbd(peps,Dnew,noise=noise)

    # Figure out peps size
    Nx = len(peps)
    Ny = len(peps[0])

    for col in range(Nx):
        for row in range(Ny):

            # Determine tensor shape
            old_shape = list(peps[row][col].ten.shape)
            new_shape = list(peps[row][col].ten.shape)
            legs = peps[row][col].legs

            # Determine new required shape
            ind = tuple()
            # Left bond
            if row != 0:
                new_shape[legs[0][0]] = Dnew
                ind += (slice(0,old_shape[legs[0][0]]),)
            else:
                for i in legs[0]:
                    ind += (slice(0,old_shape[i]),)
            # Down Bond
            if col != 0:
                new_shape[peps[row][col].legs[1][0]] = Dnew
                ind += (slice(0,old_shape[legs[1][0]]),)
            else:
                for i in legs[1]:
                    ind += (slice(0,old_shape[i]),)
            # Physical Bond
            for i in legs[2]:
                ind += (slice(0,old_shape[i]),)
            # Right Bond
            if row != Nx-1:
                new_shape[peps[row][col].legs[3][0]] = Dnew
                ind += (slice(0,old_shape[legs[3][0]]),)
            else:
                for i in legs[3]:
                    ind += (slice(0,old_shape[i]),)
            # Top Bond
            if col != Ny-1:
                new_shape[peps[row][col].legs[4][0]] = Dnew
                ind += (slice(0,old_shape[legs[4][0]]),)
            else:
                for i in legs[4]:
                    ind += (slice(0,old_shape[i]),)

            # Create an empty tensor
            ten = peps.backend.zeros(new_shape,dtype=peps[row][col].dtype)
            ten[ind] = peps[row][col].ten.copy()

            # Add some noise (if needed
            ten_noise = noise*peps.backend.random(new_shape)
            ten += ten_noise

            # Put new tensor back into peps
            peps[row][col].ten = ten

    # Increase Lambda tensors as well if needed
    peps = increase_peps_mbd_lambda(peps,Dnew,noise=noise) 

    # Normalize if needed
    if normalize: 
        peps.normalize(chi=chi)

    # Return result
    return peps

def copy_peps_tensors(peps):
    """
    Create a copy of the PEPS tensors
    """
    cp = []
    for x in range(len(peps)):
        tmp = []
        for y in range(len(peps[0])):
            tmp += [peps[x][y].copy()]
        cp += [tmp]
    return cp

def copy_lambda_tensors(peps):
    """
    Create a copy of the PEPS tensors
    """
    cp = []
    for x in range(len(peps.ltensors)):
        tmp = []
        for y in range(len(peps.ltensors[x])):
            tmp2 = []
            for z in range(len(peps.ltensors[x][y])):
                tmp2 += [peps.ltensors[x][y][z].copy()]
            tmp += [tmp2]
        cp += [tmp]
    return cp

def peps_absorb_lambdas(Gamma,Lambda,mk_copy=True):
    """
    Absorb the lambda tensors into the gamma tensors,
    transforming the peps representations from the canonical
    Gamma-Lambda form into the standard representation.

    Args:
        Gamma : list of lists
            A list of a list of the peps gamma tensors
        Lambda : list of lists of lists
            The lambda tensors (singular value vectors)
            with Lambda[0] being the lambda vecs on the vertical bonds and
            Lambda[1] being the lambda vecs on the horizontal bonds.

    Returns:
        peps : list of lists
            The peps tensors
    """

    if Lambda is not None:
        # Create a copy of Gamma (if needed)
        if mk_copy:
            Gamma = copy_peps_tensors(Gamma)

        # Figure out peps lattice size
        Nx = len(Gamma)
        Ny = len(Gamma[0])

        # loop through all sites, absorbing the "singular values"
        for x in range(Nx):
            for y in range(Ny):
                # Absorb lambdas that are to the right and above (not symmetric
                # but better for precision)
                initsgn = Gamma[x][y].get_signs()
                if x is not Nx-1:
                    Gamma[x][y] = einsum('ldpru,rR->ldpRu',Gamma[x][y],Lambda[1][x][y])
                if y is not Ny-1:
                    Gamma[x][y] = einsum('ldpru,uU->ldprU',Gamma[x][y],Lambda[0][x][y])
                Gamma[x][y].update_signs(initsgn)
    # Return results
    return Gamma

def load_peps(fname):
    """
    Load a saved PEPS into a new PEPS object

    Args:
        fname : str
            The file which holds the saved PEPS object

    Returns:
        peps : PEPS object
            A peps object with the saved PEPS loaded
    """
    # Open File
    f = open_file(fname,'r')

    # Get PEPS info
    Nx = get_dataset('Nx')
    Ny = get_dataset('Ny')
    shape = get_dataset('shape')
    d = get_dataset('d')
    D = get_dataset('D')
    chi = get_dataset('chi')
    norm_tol = get_dataset('norm_tol')
    canonical = get_dataset('canonical')
    singleLayer = get_dataset('singleLayer')
    max_norm_iter = get_dataset('max_norm_iter')
    norm_BS_upper = get_dataset('norm_BS_upper')
    norm_BS_lower = get_dataset('norm_BS_lower')
    norm_BS_print = get_dataset('norm_BS_print')
    dtype = get_dataset('tensor_0_0').dtype
    fname = get_dataset('fname')
    fdir = get_dataset('fdir')

    # Create new PEPS object
    peps = PEPS(Nx=Nx,Ny=Ny,d=d,D=D,
                chi=chi,norm_tol=norm_tol,
                canonical=canonical,
                singleLayer=singleLayer,
                max_norm_iter=max_norm_iter,
                norm_BS_upper=norm_BS_upper,
                norm_BS_lower=norm_BS_lower,
                dtype=dtype,normalize=False,
                fdir=fdir,fname=fname+'_loaded')

    # Load PEPS Tensors
    for i in range(Nx):
        for j in range(Ny):
            peps.tensors[i][j] = get_dataset('tensor_{}_{}'.format(i,j))

    # Load lambda tensors (if there)
    if canonical:
        for ind in range(len(self.ltensors)):
            for x in range(len(self.ltensors[ind])):
                for y in range(len(self.ltensors[ind][x])):
                    peps.ltensors[ind][x][y] = get_dataset('ltensor_{}_{}_{}'.format(ind,x,y))

    # Return resulting PEPS
    return peps

# -----------------------------------------------------------------
# PEPS Class

class PEPS:
    """
    A class to hold and manipulate a PEPS
    """

    def __init__(self,Nx=10,Ny=10,d=2,D=2,
                 chi=None,Zn=None,thermal=False,
                 dZn=None,canonical=False,backend='numpy',
                 singleLayer=True,dtype=float_,
                 normalize=True,norm_tol=20.,
                 max_norm_iter=100,norm_bs_upper=10.0,norm_bs_lower=0.0,
                 fname=None,fdir='./'):
        """
        Create a random PEPS object

        Args:
            self : PEPS Object

        Kwargs:
            Nx : int
                The length of the lattice in the x-direction
            Ny : int
                The length of the lattice in the y-direction
            d : int
                The local bond dimension
            D : int
                The auxilliary bond dimension
            chi : int
                The boundary mpo maximum bond dimension
            Zn : int
                Create a PEPS which preserves this Zn symmetry,
                i.e. if Zn=2, then Z2 symmetry is preserved.
            thermal : bool
                Whether or not to create a thermal state,
                i.e. an additional physical index
            dZn : int
                The number of symmetry sectors in the physical
                bond dimension. 
            canonical : bool
                If true, then the PEPS will be created in the 
                Gamma Lambda formalism, with diagonal matrices
                between each set of PEPS tensors. Default is False,
                where a standard PEPS, with one tensor per site,
                will be created.
            backend : str
                This specifies the backend to be used for the calculation.
                Options are currently 'numpy' or 'ctf'. If using symmetries,
                this will be adapted to using symtensors with numpy or ctf as
                the backend.
            singleLayer : bool
                Whether to use a single layer environment
                (currently only option implemented)
            norm_tol : float
                How close to 1. the norm should be before exact
                artihmetic is used in the normalization procedure.
                See documentation of normalize_peps() function
                for more details.
            dtype : dtype
                The data type for the PEPS
            normalize : bool
                Whether the initialized random peps should be normalized
            max_norm_iter : int
                The maximum number of normalization iterations
            norm_bs_upper : float
                The upper bound for the binary search factor
                during normalization.
            norm_bs_lower : float
                The lower bound for the binary search factor
                during normalization.
            norm_BS_print : boolean
                Controls output of binary search normalization
                procedure.
            dtype : dtype
                The data type for the PEPS
            normalize : bool
                Whether the initial random peps should be normalized
            fname : str
                Where the PEPS will be saved as an .npz file, if None,
                then the default is 'peps_Nx{}_Ny{}_D{}'
            fdir : str
                The directory where the PEPS will be saved, default is
                current working directory

        Returns:
            PEPS : PEPS Object
                The resulting random projected entangled pair
                state as a PEPS object
        """
        # Collect input arguments
        self.Nx          = Nx
        self.Ny          = Ny
        self.shape       = (Nx,Ny)
        self.d           = d
        self.D           = D
        if chi is None: chi = 4*D**2
        self.chi         = chi
        self.Zn          = Zn
        self.thermal     = thermal
        if dZn is None: dZn = Zn
        self.dZn         = dZn
        self.canonical   = canonical
        self.backend     = backend
        self.backend     = load_lib(self.backend)
        self.singleLayer = singleLayer
        self.dtype       = dtype
        self.norm_tol    = norm_tol
        self.max_norm_iter = max_norm_iter
        self.norm_bs_upper = norm_bs_upper
        self.norm_bs_lower = norm_bs_lower
        if fname is None:
            self.fname = 'peps_Nx{}_Ny{}_D{}'.format(Nx,Ny,D)
        else:
            self.fname = fname
        self.fdir          = fdir

        # Make a random PEPS
        if thermal:
            self.tensors = make_thermal_peps(self.Nx,
                                             self.Ny,
                                             self.d,
                                             self.D,
                                             Zn=self.Zn,
                                             dZn=self.dZn,
                                             backend=self.backend,
                                             dtype=self.dtype)
        else:
            self.tensors = make_rand_peps(self.Nx,
                                          self.Ny,
                                          self.d,
                                          self.D,
                                          Zn=self.Zn,
                                          dZn=self.dZn,
                                          backend=self.backend,
                                          dtype=self.dtype)

        # Add in lambda "singular value" matrices
        if self.canonical:
            if thermal:
                self.ltensors = make_thermal_lambdas(self.Nx,
                                                     self.Ny,
                                                     self.D,
                                                     Zn=self.Zn,
                                                     backend=self.backend,
                                                     dtype=self.dtype)
            else:
                self.ltensors = make_rand_lambdas(self.Nx,
                                                  self.Ny,
                                                  self.D,
                                                  Zn=self.Zn,
                                                  backend=self.backend,
                                                  dtype=self.dtype)
        else:
            self.ltensors = None

        # Normalize the PEPS
        if normalize:
            self.normalize()

    def calc_bmpo_left(self,col,chi=4,singleLayer=True,truncate=True,return_all=False,ket=None):
        """
        Calculate the left boundary MPO

        Args:
            peps : List
                A list of lists containing the peps tensors
            col : int
                The last column for which you need the environment

        Kwargs:
            chi : int
                The maximum bond dimension of the boundary MPO
            single_layer : bool
                Indicates whether to use a single layer environment
                (currently it is the only option...)
            truncate : bool
                Whether or not to do an svd and truncate the resulting
                boundary mpo
            return_all : bool
                Whether to return a list of boundary mpos upto col or just
                return the boundary mpo for col.
            ket : List
                A list of lists containing the ket's peps tensors

        returns:
            bound_mpo : list
                An mpo stored as a list, corresponding to the
                resulting boundary mpo.
        """
        if chi is None:
            chi = self.chi
        if singleLayer is None:
            singleLayer = self.singleLayer
        return calc_left_bound_mpo(self,col,chi=chi,singleLayer=singleLayer,truncate=truncate,return_all=return_all,ket=ket)

    def calc_bmpo_right(self,col,chi=None,singleLayer=None,truncate=True,return_all=False,ket=None):
        """
        Calculate the right boundary MPO

        Args:
            peps : List
                A list of lists containing the peps tensors
            col : int or list of ints
                The column(s) for which you need the environment

        Kwargs:
            chi : int
                The maximum bond dimension of the boundary MPO
            single_layer : bool
                Indicates whether to use a single layer environment
                (currently it is the only option...)
            truncate : bool
                Whether or not to do an svd and truncate the resulting
                boundary mpo
            return_all : bool
                Whether to return a list of boundary mpos upto col or just
                return the boundary mpo for col.
            ket : List
                A list of lists containing the ket's peps tensors

        returns:
            bound_mpo : list
                An mpo stored as a list, corresponding to the
                resulting boundary mpo.

        """
        if chi is None:
            chi = self.chi
        if singleLayer is None:
            singleLayer = self.singleLayer
        return calc_right_bound_mpo(self,col,chi=chi,singleLayer=singleLayer,truncate=truncate,return_all=return_all,ket=ket)

    def calc_norm(self,chi=None,singleLayer=None,ket=None):
        """
        Calculate the norm of the PEPS

        Args:
            self : PEPS Object

        Kwargs:
            chi : int
                The boundary MPO bond dimension
            single_layer : bool
                Indicates whether to use a single layer environment
                (currently it is the only option...)
            ket : PEPS Object
                A peps containing the ket's peps tensors

        Returns:
            norm : float
                The (approximate) norm of the PEPS
        """
        if chi is None: chi = self.chi
        if singleLayer is None: singleLayer = self.singleLayer
        return calc_peps_norm(self,chi=chi,singleLayer=singleLayer,ket=ket)

    def normalize(self,max_iter=None,norm_tol=None,chi=None,up=None,down=None,
                    singleLayer=None):
        """
        Normalize the full PEPS

        Args:
            self : PEPS Object
                The PEPS to be normalized

        Kwargs:
            max_iter : int
                The maximum number of iterations of the normalization
                procedure. Default is 20.
            norm_tol : int
                We require the measured norm to be within the bounds
                10^(-norm_tol) < norm < 10^(norm_tol) before we do
                exact arithmetic to get the norm very close to 1. Default
                is 20.
            chi : int
                Boundary MPO maximum bond dimension
            up : float
                The upper bound for the binary search factor. Default is 1.0,
                which assumes that the norm of the initial PEPS is greater
                than 1 (this is almost always true).
            down : float
                The lower bound for the binary search factor. Default is 0.0.
                The intial guess for the scale factor is the midpoint
                between up and down. It's not recommended to adjust the
                up and down parameters unless you really understand what
                they are doing.
            single_layer : bool
                Indicates whether to use a single layer environment
                (currently it is the only option...)

        Returns:
            norm : float
                The approximate norm of the PEPS after the normalization
                procedure
        """
        # Figure out good chi (if not given)
        if chi is None: chi = self.chi
        if max_iter is None: max_iter = self.max_norm_iter
        if norm_tol is None: norm_tol = self.norm_tol
        if up is None: up = self.norm_bs_upper
        if down is None: down = self.norm_bs_lower
        if singleLayer is None: singleLayer = self.singleLayer
        # Run the normalization procedure
        norm, normpeps = normalize_peps(self,
                                      max_iter = max_iter,
                                      norm_tol = norm_tol,
                                      chi = chi,
                                      up = up,
                                      down = down,
                                      singleLayer=singleLayer)
        # Copy the resulting tensors
        self.tensors = copy_peps_tensors(normpeps)
        if self.ltensors is not None:
            self.ltensors = copy_lambda_tensors(normpeps)

        return norm

    def calc_op(self,ops,chi=None,normalize=True,return_sum=True,ket=None,nn=False,contracted_env=False):
        """
        Calculate the expectation value for a given operator

        Args:
            self : PEPS Object
                The PEPS to be normalized
            ops :
                The operator to be contracted with the peps

        Kwargs:
            chi : int
                The maximum bond dimension for the boundary mpo
            normalize : bool
                Whether to divide the resulting operator value by the peps norm
            return_sum : bool
                Whether to either return an array of the results, the same shape
                as ops, or a summation of all operators
            ket : PEPS Object
                A second peps, to use as the ket, in the operator contraction
            nn : bool
                Whether the Hamiltonian involves next nearest (nn) neighbor interactions
            contracted_env: bool
                Whether to contract the upper and lower environment
                or leave it as a boundary mps

        Returns:
            val : float
                The resulting observable's expectation value
        """
        if chi is None: chi = self.chi
        # Calculate the operator's value
        if nn:
            return calc_peps_nn_op(self,ops,chi=chi,normalize=normalize,ket=ket,contracted_env=contracted_env)
        else:
            return calc_peps_op(self,ops,chi=chi,normalize=normalize,return_sum=return_sum,ket=ket)

    def increase_mbd(self,newD,chi=None,noise=1e-10,normalize=True):
        """
        Increase the maximum bond dimension of the peps

        Args:
            self : PEPS Object
                The PEPS to be normalized

        Kwargs:
            new_chi : int
                The new bond dimension for the boundary mpo

        """
        if newD is not None:
            self.chi = chi
        self = increase_peps_mbd(self,newD,noise=noise,normalize=normalize,chi=chi)

    def absorb_lambdas(self):
        """
        Absorb the lambda from the canonical Gamma-Lambda PEPS
        form to return the normal PEPS form.

        Args:
            self : PEPS Object
                The PEPS to be normalized
        """
        self.tensors = peps_absorb_lambdas(self.tensors,self.ltensors)
        self.ltensors = None
        self.canonical = False

    def __len__(self):
        return self.Nx

    def __getitem__(self,ind):
        return self.tensors[ind]

    def __setitem__(self,ind,item):
        self.tensors[ind] = item

    def copy(self):
        """
        Return a copy of this PEPS
        """
        peps_copy = PEPS(Nx=self.Nx,Ny=self.Ny,d=self.d,D=self.D,
                         chi=self.chi,norm_tol=self.norm_tol,
                         canonical=self.canonical,
                         singleLayer=self.singleLayer,
                         backend=self.backend,
                         max_norm_iter=self.max_norm_iter,
                         norm_bs_upper=self.norm_bs_upper,
                         norm_bs_lower=self.norm_bs_lower,
                         dtype=self.dtype,normalize=False,
                         fdir=self.fdir,fname=self.fname+'_cp')

        # Copy peps tensors
        for i in range(self.Nx):
            for j in range(self.Ny):
                peps_copy.tensors[i][j] = self.tensors[i][j].copy()

        # Copy lambda tensors (if there)
        if self.ltensors is not None:
            for ind in range(len(self.ltensors)):
                for x in range(len(self.ltensors[ind])):
                    for y in range(len(self.ltensors[ind][x])):
                        peps_copy.ltensors[ind][x][y] = self.ltensors[ind][x][y].copy()

        # Return result
        return peps_copy

    def rotate(self,clockwise=True):
        """
        Rotate the peps

        Args:
            peps : a list of a list containing peps tensors
                The initial peps tensor

        Kwargs:
            clockwise : bool
                Rotates clockwise if True, counter-clockwise
                otherwise

        """
        self.tensors = rotate_peps(self.tensors,clockwise=clockwise)
        self.ltensors= rotate_lambda(self.ltensors,clockwise=clockwise)
        Nx_ = self.Nx
        Ny_ = self.Ny
        self.Nx = Ny_
        self.Ny = Nx_
        self.shape = (self.Nx,self.Ny)

    def flip(self):
        """
        Flip the peps columns
        """
        self.tensors = flip_peps(self.tensors)
        self.ltensors= flip_lambda(self.ltensors)

    def make_sparse(self):
        """
        Convert the densely stored symmetric PEPS to a sparsely stored symmetric PEPS
        """
        # Create the new peps objects
        speps = PEPS(Nx            = self.Nx,
                     Ny            = self.Ny,
                     d             = self.d,
                     D             = self.D,
                     chi           = self.chi,
                     Zn            = None,
                     canonical     = self.canonical,
                     backend       = self.backend,
                     singleLayer   = self.singleLayer,
                     dtype         = self.dtype,
                     norm_tol      = self.norm_tol,
                     max_norm_iter = self.max_norm_iter,
                     norm_bs_upper = self.norm_bs_upper,
                     norm_bs_lower = self.norm_bs_lower,
                     normalize=False)

        # Loop through all sites converting tensors to sparse
        for x in range(speps.Nx):
            for y in range(speps.Ny):
                # Get a sparse version of the tensors
                speps[x][y] = self.tensors[x][y].copy().make_sparse()
        # Do it for lambda tensors as well
        if speps.canonical:
            for x in range(len(speps.ltensors)):
                for y in range(len(speps.ltensors[x])):
                    for z in range(len(speps.ltensors[x][y])):
                        speps.ltensors[x][y][z] = self.ltensors[x][y][z].copy().make_sparse()
        # Return result
        return speps

    def save(self):
        """
        Save the PEPS tensors
        """
        # Create file
        f = open_file(self.fdir+self.fname,'w')
        # Add PEPS Info
        create_dataset(f,'Nx',self.Nx)
        create_dataset(f,'Ny',self.Ny)
        create_dataset(f,'shape',self.shape)
        create_dataset(f,'d',self.d)
        create_dataset(f,'D',self.D)
        create_dataset(f,'chi',self.chi)
        create_dataset(f,'norm_tol',self.norm_tol)
        create_dataset(f,'canonical',self.canonical)
        create_dataset(f,'singleLayer',self.singleLayer)
        create_dataset(f,'max_norm_iter',self.max_norm_iter)
        create_dataset(f,'norm_BS_upper',self.norm_BS_upper)
        create_dataset(f,'norm_BS_lower',self.norm_BS_lower)
        create_dataset(f,'norm_BS_print',self.norm_BS_print)
        #create_dataset(f,'dtype',self.dtype) # NOTE - Not able to save dtype...
        create_dataset(f,'fname',self.fname)
        create_dataset(f,'fdir',self.fdir)
        # Add PEPS Tensors
        for i in range(len(self.tensors)):
            for j in range(len(self.tensors[i])):
                create_dataset(f,'tensor_{}_{}'.format(i,j),self.tensors[i][j])
        # Add Lambda Tensors (if Canonical)
        if self.ltensors is not None:
            for ind in range(len(self.ltensors)):
                for x in range(len(self.ltensors[ind])):
                    for y in range(len(self.ltensors[ind][x])):
                        create_dataset(f,'ltensor_{}_{}_{}'.format(ind,x,y),self.ltensors[ind][x][y])
        # Close file
        close_file(f)
