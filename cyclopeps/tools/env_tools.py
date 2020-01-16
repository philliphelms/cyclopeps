"""
Tools for the environment (or boundary MPOs) of
a projected entangled pair states

Author: Phillip Helms <phelms@caltech.edu>
Date: July 2019

"""

from cyclopeps.tools.gen_ten import einsum,eye
from cyclopeps.tools.utils import *
from cyclopeps.tools.mps_tools import MPS
from cyclopeps.tools.peps_tools import *
from numpy import float_
import copy

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

def init_left_bmpo_sl(ket, bra=None, chi=4, truncate=True):
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

    Returns:
        bound_mpo : list
            An updated boundary mpo
    """
    mpiprint(3,'Initial Layer of left boundary mpo (sl)')
    # Find size of peps column and dims of tensors
    Ny = len(ket)
    _,_,d,D,_ = ket[0].shape

    # Copy the bra and ket
    ket = copy_tensor_list(ket)
    if bra is None:
        bra = copy_tensor_list(ket)

    # Make list to hold resulting mpo
    bound_mpo = []

    for row in range(Ny):
        # Add Bra-ket contraction ------------------------------
        res = einsum('ldpru,LDpRU->lLdDRurU',ket[row],bra[row])
        # Merge inds to make it an MPO
        res.merge_inds([0,1,2,3])
        res.merge_inds([2,3,4])
        # Append to boundary_mpo
        bound_mpo.append(res)

        # Add correct identity ---------------------------------
        (Dl,Dd,Dp,Dr,Du) = ket[row].shape
        (Zl,Zd,Zp,Zr,Zu) = ket[row].qn_sectors
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
        Itmp = einsum('du,DU->dDuU',I1,I2)
        I = einsum('dDuU,lr->dlDruU',Itmp,I3)
        # Merge inds to make it an MPO
        I.merge_inds([0,1,2])
        I.merge_inds([2,3])
        # Append to the boundary mpo
        bound_mpo.append(I)

    # Put result into an MPS -------------------------------------------
    bound_mps = MPS(bound_mpo)

    # Reduce bond dimension
    if truncate:
        print('Here?')
        norm0 = bound_mps.norm()
        bound_mps.apply_svd(chi)
        norm1 = bound_mps.norm()
        mpiprint(4,'Norm Difference for chi={}: {}'.format(chi,abs(norm0-norm1)/abs(norm0)))
    return bound_mps

def left_bmpo_sl_add_ket(ket,bound_mpo,D,Ny,chi=4,truncate=True):
    """
    Add the ket layer to the boundary mpo
    """
    mpiprint(4,'Adding Ket')

    # Make list to hold resulting mpo
    bound_mpo_new = []

    for row in range(Ny):
        mpiprint(5,'Adding Site {} to Ket'.format(Ny))

        # Add correct identity
        mpiprint(6,'Adding Identity to ket boundary mps')
        (Dl,Dd,Dp,Dr,Du) = ket[row].shape
        I1 = eye(Dd)
        I = einsum('mLn,du->mdLnu',bound_mpo[2*row],I1)
        # Reshape it into an MPO
        (Dm,DL,Dn) = bound_mpo[2*row].shape
        I = reshape(I,(Dm*Dd,DL,Dn*Dd))
        # Append to new boundary mpo
        bound_mpo_new.append(I)

        # Add ket contraction
        mpiprint(6,'Adding Identity ket tensor to boundary mps')
        res = einsum('mln,ldpru->mdrpnu',bound_mpo[2*row+1],ket[row])
        # Reshape it into an MPO
        (Dm,_,Dn) = bound_mpo[2*row+1].shape
        res = reshape(res,(Dm*Dd,Dr*Dp,Dn*Du))
        # Append to new boundary mpo
        bound_mpo_new.append(res)

    # Put result into an MPS -------------------------
    mpiprint(7,'Putting MPS list into MPS object')
    bound_mps = MPS()
    bound_mps.input_mps_list(bound_mpo_new)

    # Reduce bond dimension
    if truncate:
        mpiprint(5,'Truncating Boundary MPS')
        #mpiprint(6,'Computing initial norm')
        #norm0 = bound_mps.norm()
        #mpiprint(6,'Applying SVD')
        bound_mps.apply_svd(chi)
        #mpiprint(6,'Computing Resulting norm')
        #norm1 = bound_mps.norm()
        #mpiprint(4,'Norm Difference for chi={}: {}'.format(chi,abs(norm0-norm1)/abs(norm0)))
    return bound_mps

def left_bmpo_sl_add_bra(bra,bound_mpo,D,Ny,chi=4,truncate=True):
    """
    Add the bra layer to the boundary mpo
    """
    mpiprint(4,'Adding Bra')
    # Make list to hold resulting mpo
    bound_mpo_new = []

    for row in range(Ny):
        mpiprint(5,'Adding Site {} to bra'.format(Ny))

        # Add bra contraction
        mpiprint(6,'Adding bra tensor to boundary mps')
        res = einsum('mLn,LDPRU->mDRnUP',bound_mpo[2*row],bra[row])
        # Reshape it into an MPO
        (Dm,_,Dn) = bound_mpo[2*row].shape
        (DL,DD,DP,DR,DU) = bra[row].shape
        res = reshape(res,(Dm*DD,DR,Dn*DU*DP))
        # Append to new boundary MPO
        bound_mpo_new.append(res)

        # Add correct identity
        mpiprint(6,'Adding Identity boundary mps')
        bound_tens = bound_mpo[2*row+1]
        (Dm,Dp,Dn) = bound_tens.shape
        d = int(Dp/D)
        bound_tens = reshape(bound_tens,(Dm,D,d,Dn))
        I = eye(DU)
        # Contract with previous bound_mpo tensor
        res = einsum('mrpn,DU->mDprnU',bound_tens,I)
        # Reshape it back into an MPO
        res = reshape(res,(Dm*DU*d,D,Dn*DU))
        # Append to new boundary MPO
        bound_mpo_new.append(res)

    # Put result into an MPS -------------------------
    bound_mps = MPS()
    bound_mps.input_mps_list(bound_mpo_new)

    # Reduce bond dimension
    mpiprint(7,'Putting MPS list into MPS object')
    if truncate:
        mpiprint(5,'Truncating Boundary MPS')
        #mpiprint(6,'Computing initial norm')
        #norm0 = bound_mps.norm()
        #mpiprint(6,'Applying SVD')
        bound_mps.apply_svd(chi)
        #mpiprint(6,'Computing Resulting norm')
        #norm1 = bound_mps.norm()
        #mpiprint(4,'Norm Difference for chi={}: {}'.format(chi,abs(norm0-norm1)/abs(norm0)))
    return bound_mps

def left_bmpo_sl(ket, bound_mpo, bra=None, chi=4,truncate=True):
    """
    Add two layers to the single layer boundary mpo environment

    Args:
        ket : list
            A list containing the tensors for a single peps column
        bound_mpo : list
            A list containing the tensors for the left neighboring
            boundary mpo

    Kwargs:
        bra : list
            A list containing the bra tensors for a single peps column. 
            If None, then the ket col will be used
        chi : int
            The maximum bond dimension for the boundary mpo
        truncate : bool
            Whether or not to do an svd and truncate the resulting
            boundary mpo

    Returns:
        bound_mpo : list
            An updated boundary mpo
    """
    mpiprint(5,'Updating boundary mpo (sl)')
    # Find size of peps column and dims of tensors
    Ny = len(ket)
    _,_,d,D,_ = ket[0].shape
    
    # Copy the ket column if needed
    if bra is None:
        bra = ket.copy()

    # First Layer (ket) #####################################
    bound_mpo = left_bmpo_sl_add_ket(ket,bound_mpo,D,Ny,chi=chi,truncate=truncate)
    # Second Layer (bra) ####################################
    bound_mpo = left_bmpo_sl_add_bra(bra,bound_mpo,D,Ny,chi=chi,truncate=truncate)

    # Return result
    return bound_mpo

def left_update_sl(peps_col, bound_mpo, chi=4,truncate=True):
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

    Returns:
        bound_mpo : list
            An updated boundary mpo
    """
    # Check if we are at left edge
    if bound_mpo is None:
        bound_mpo = init_left_bmpo_sl(peps_col,chi=chi,truncate=truncate)
    # Otherwise update is generic
    else:
        # Start from bottom of the column
        bound_mpo = left_bmpo_sl(peps_col,bound_mpo,chi=chi,truncate=truncate)
    return bound_mpo

def left_update(peps_col,bound_mpo,chi=4):
    mpiprint(0,'Only single layer environment implemented')
    import sys
    sys.exit()

def update_left_bound_mpo(peps_col, bound_mpo, chi=4, singleLayer=True,truncate=True):
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

    Returns:
        bound_mpo : list
            An updated boundary mpo
    """
    if singleLayer:
        return left_update_sl(peps_col,bound_mpo,chi=chi,truncate=truncate)
    else:
        return left_update(peps_col,bound_mpo,chi=chi,truncate=truncate)

def calc_left_bound_mpo(peps,col,chi=4,singleLayer=True,truncate=True,return_all=False):
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
        if colind == 0:
            bound_mpo[colind] = update_left_bound_mpo(peps[colind][:], None, chi=chi, singleLayer=singleLayer,truncate=truncate)
        else:
            bound_mpo[colind] = update_left_bound_mpo(peps[colind][:], bound_mpo[colind-1], chi=chi, singleLayer=singleLayer,truncate=truncate)

    # Return result
    if return_all:
        return bound_mpo
    else:
        return bound_mpo[-1]

def calc_right_bound_mpo(peps,col,chi=4,singleLayer=True,truncate=True,return_all=False):
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
    col = Nx-col

    # Loop through the columns, creating a boundary mpo for each
    bound_mpo = [None]*(col-1)
    for colind in range(col-1):
        mpiprint(4,'Updating boundary mpo')
        if colind == 0:
            bound_mpo[colind] = update_left_bound_mpo(peps[colind][:], None, chi=chi, singleLayer=singleLayer, truncate=truncate)
        else:
            bound_mpo[colind] = update_left_bound_mpo(peps[colind][:], bound_mpo[colind-1], chi=chi, singleLayer=singleLayer, truncate=truncate)

    # Return results
    if return_all:
        return bound_mpo[::-1]
    else:
        return bound_mpo[-1]
