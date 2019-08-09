"""
Tools for the environment (or boundary MPOs) of
a projected entangled pair states

Author: Phillip Helms <phelms@caltech.edu>
Date: July 2019

"""

from cyclopeps.tools.utils import *
from cyclopeps.tools.mps_tools import MPS
from numpy import float_
import peps_tools 
import copy

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PEPS ENVIRONMENT FUNCTIONS 
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def init_left_bmpo_sl(peps_col,chi=4,truncate=True):
    """
    Create the initial boundary mpo for a peps

    Args:
        peps_col : list
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
    Ny = len(peps_col)
    _,_,d,D,_ = peps_col[0].shape

    # get the bra and ket as copies of the peps column
    bra = [conj(copy.copy(peps_col[i])) for i in range(len(peps_col))]
    ket = [copy.copy(peps_col[i]) for i in range(len(peps_col))]

    # Make list to hold resulting mpo
    bound_mpo = []

    # Bottom row ---------------------------------------------------------
    # Add bra-ket contraction
    res = einsum('ldpru,LDpRU->lLdDRurU',ket[0],bra[0])
    # Reshape so it is an MPO
    res = reshape(res,(1,D,D*D*D))
    # Append to boundary_mpo
    bound_mpo.append(res)

    # Add correct identity
    I = eye(D)
    I = einsum('du,DU,lr->dlDruU',I,I,I)
    I = reshape(I,(D*D*D,D,D*D))
    # Append to boundary mpo
    bound_mpo.append(I)

    # Central Rows ------------------------------------------------------
    for row in range(1,Ny-1):
        # Add bra-ket contraction
        res = einsum('ldpru,LDpRU->lLdDRurU',ket[row],bra[row])
        # Reshape it into an MPO
        res = reshape(res,(D*D,D,D*D*D))
        # Append to boundary mpo
        bound_mpo.append(res)

        # Add correct identity
        I = eye(D)
        I = einsum('du,lr,DU->dlDruU',I,I,I)
        I = reshape(I,(D*D*D,D,D*D))
        # Append to boundary mpo
        bound_mpo.append(I)

    # Top Row ----------------------------------------------------------
    # Add Correct identity first
    I = eye(D)
    I = einsum('du,lr,DU->dDrulU',I,I,I)
    I = reshape(I,(D*D,D,D*D*D))
    # Append to boundary mpo
    bound_mpo.append(I)

    # Add bra-ket contraction
    res = einsum('ldpru,LDpRU->lLdrDRuU',ket[-1],bra[-1])
    # Reshape into an MPO
    res = reshape(res,(D*D*D,D,1))
    # Append to boundary mpo
    bound_mpo.append(res)

    # Put result into an MPS -------------------------------------------
    bound_mps = MPS()
    bound_mps.input_mps_list(bound_mpo)

    # Reduce bond dimension
    if truncate:
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

    # Bottom row -----------------------------------
    # Add previous first site
    bound_mpo_new.append(copy.copy(bound_mpo[0]))

    # Add ket contraction
    res = einsum('mLn,LDPRU->mDRPnU',bound_mpo[1],ket[0])
    # Reshape it into an MPO
    (Dm,DL,Dn) = bound_mpo[1].shape
    (DL,DD,DP,DR,DU) = ket[0].shape
    res = reshape(res,(Dm*DD,DR*DP,Dn*DU))
    # Append to new boundary mpo
    bound_mpo_new.append(res)

    # Add correct identity
    I = eye(D)
    I = einsum('mLn,LR,DU->mDRnU',bound_mpo[2],I,I)
    # Reshape it into an MPO
    (Dm,DL,Dn) = bound_mpo[2].shape
    I = reshape(I,(Dm*D,D,Dn*D))
    # Append to new boundary mpo
    bound_mpo_new.append(I)
    
    # Center Rows ----------------------------------
    for row in range(1,Ny-2):
        # Add ket
        res = einsum('mLn,LDPRU->mDRPnU',bound_mpo[2*row+1],ket[row])
        # Reshape it into an MPO
        (Dm,DL,Dn) = bound_mpo[2*row+1].shape
        (DL,DD,DP,DR,DU) = ket[row].shape
        res = reshape(res,(Dm*DD,DR*DP,Dn*DU))
        # Append to new boundary mpo
        bound_mpo_new.append(res)

        # Add correct identity
        I = eye(D)
        I = einsum('mLn,LR,DU->mDRnU',bound_mpo[2*row+2],I,I)
        # Reshape it into an MPO
        (Dm,DL,Dn) = bound_mpo[2*row+2].shape
        I = reshape(I,(Dm*D,D,Dn*D))
        # Append to new boundary mpo
        bound_mpo_new.append(I)

    # Top Row -------------------------------------
    # Add ket
    res = einsum('mLn,LDPRU->mDRPnU',bound_mpo[-3],ket[-2])
    # Reshape it into an MPO
    (Dm,DL,Dn) = bound_mpo[-3].shape
    (DL,DD,DP,DR,DU) = ket[-2].shape
    res = reshape(res,(Dm*DD,DR*DP,Dn*DU))
    # Append to new boundary mpo
    bound_mpo_new.append(res)

    # Add ket again
    res = einsum('mLn,LDPRU->mDRPnU',bound_mpo[-2],ket[-1])
    # Reshape it into an MPO
    (Dm,DL,Dn) = bound_mpo[-2].shape
    (DL,DD,DP,DR,DU) = ket[-1].shape
    res = reshape(res,(Dm*DD,DR*DP,Dn*DU))
    # Append to new boundary mpo
    bound_mpo_new.append(res)

    # Add previous last site
    bound_mpo_new.append(copy.copy(bound_mpo[-1]))

    # Put result into an MPS -------------------------
    bound_mps = MPS()
    bound_mps.input_mps_list(bound_mpo_new)

    # Reduce bond dimension
    if truncate:
        norm0 = bound_mps.norm()
        bound_mps.apply_svd(chi)
        norm1 = bound_mps.norm()
        mpiprint(4,'Norm Difference for chi={}: {}'.format(chi,abs(norm0-norm1)/abs(norm0)))
    return bound_mps

def left_bmpo_sl_add_bra(bra,bound_mpo,D,Ny,chi=4,truncate=True):
    """
    Add the bra layer to the boundary mpo
    """
    mpiprint(4,'Adding Bra')
    # Make list to hold resulting mpo
    bound_mpo_new = []

    # Bottom row -----------------------------------
    # Add bra contraction
    res = einsum('mLn,LDPRU->mDRnUP',bound_mpo[0],bra[0])
    # Reshape it into an MPO
    (Dm,DL,Dn) = bound_mpo[0].shape
    (DL,DD,DP,DR,DU) = bra[0].shape
    res = reshape(res,(Dm*DD,DR,Dn*DU*DP))
    # Append to new boundary mpo
    bound_mpo_new.append(res)

    # Add identity
    bound_tens = bound_mpo[1]
    (Dm,Dd,Dn) = bound_tens.shape
    d = Dd/D
    bound_tens = reshape(bound_tens,(Dm,D,d,Dn))
    I = eye(D)
    res = einsum('mLpn,DU->mDpLnU',bound_tens,I)
    # Reshape it into an MPO
    res = reshape(res,(Dm*D*d,D,Dn*D))
    # Append to new boundary MPO
    bound_mpo_new.append(res)

    # Center Rows ----------------------------------
    for row in range(1,Ny-1):
        # Add bra contraction
        res = einsum('mLn,LDPRU->mDRnUP',bound_mpo[2*row],bra[row])
        # Reshape it into an MPO
        (Dm,DL,Dn) = bound_mpo[2*row].shape
        (DL,DD,DP,DR,DU) = bra[row].shape
        res = reshape(res,(Dm*DD,DR,Dn*DU*DP))
        # Append to new boundary mpo
        bound_mpo_new.append(res)

        # Add identity
        bound_tens = bound_mpo[2*row+1]
        (Dm,Dd,Dn) = bound_tens.shape
        d = Dd/D
        bound_tens = reshape(bound_tens,(Dm,D,d,Dn))
        I = eye(D)
        res = einsum('mLpn,DU->mDpLnU',bound_tens,I)
        # Reshape it into an MPO
        res = reshape(res,(Dm*D*d,D,Dn*D))
        # Append to new boundary MPO
        bound_mpo_new.append(res)

    # Top Row -------------------------------------
    # Add identity
    bound_tens = bound_mpo[-2]
    (Dm,Dd,Dn) = bound_tens.shape
    d = Dd/D
    bound_tens = reshape(bound_tens,(Dm,D,d,Dn))
    I = eye(D)
    res = einsum('mLpn,DU->mDLnUp',bound_tens,I)
    # Reshape it into an MPO
    res = reshape(res,(Dm*D,D,Dn*D*d))
    # Append to new boundary MPO
    bound_mpo_new.append(res)

    # Add bra contraction
    res = einsum('mLn,LDPRU->mDPRnU',bound_mpo[-1],bra[-1])
    # Reshape it into an MPO
    (Dm,DL,Dn) = bound_mpo[-1].shape
    (DL,DD,DP,DR,DU) = bra[-1].shape
    res = reshape(res,(Dm*DD*DP,DR,Dn*DU))
    # Append to new boundary MPO
    bound_mpo_new.append(res)

    # Put result into an MPS -------------------------
    bound_mps = MPS()
    bound_mps.input_mps_list(bound_mpo_new)

    # Reduce bond dimension
    if truncate:
        norm0 = bound_mps.norm()
        bound_mps.apply_svd(chi)
        norm1 = bound_mps.norm()
        mpiprint(4,'Norm Difference for chi={}: {}'.format(chi,abs(norm0-norm1)/abs(norm0)))
    return bound_mps

def left_bmpo_sl(peps_col, bound_mpo, chi=4,truncate=True):
    """
    Add two layers to the single layer boundary mpo environment

    Args:
        peps_col : list
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

    Returns:
        bound_mpo : list
            An updated boundary mpo
    """
    mpiprint(5,'Updating boundary mpo (sl)')
    # Find size of peps column and dims of tensors
    Ny = len(peps_col)
    _,_,d,D,_ = peps_col[0].shape
    
    # get the bra and ket as copies of the peps column
    bra = [conj(copy.copy(peps_col[i])) for i in range(len(peps_col))]
    ket = [copy.copy(peps_col[i]) for i in range(len(peps_col))]

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

def calc_left_bound_mpo(peps,col,chi=4,singleLayer=True,truncate=True):
    """
    Calculate the left boundary MPO

    Args:
        peps : List
            A list of lists containing the peps tensors
        col : int
            The column for which you need the environment

    Kwargs:
        chi : int
            The maximum bond dimension of the boundary MPO
        single_layer : bool
            Indicates whether to use a single layer environment
            (currently it is the only option...)
        truncate : bool
            Whether or not to do an svd and truncate the resulting
            boundary mpo

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
    bound_mpo = None
    for colind in range(col-1):
        mpiprint(4,'Updating left boundary mpo')
        bound_mpo = update_left_bound_mpo(peps[colind][:], bound_mpo, chi=chi, singleLayer=singleLayer,truncate=truncate)

    # Return result
    return bound_mpo

def calc_right_bound_mpo(peps,col,chi=4,singleLayer=True,truncate=True):
    """
    Calculate the right boundary MPO

    Args:
        peps : List
            A list of lists containing the peps tensors
        col : int
            The column for which you need the environment

    Kwargs:
        chi : int
            The maximum bond dimension of the boundary MPO
        single_layer : bool
            Indicates whether to use a single layer environment
            (currently it is the only option...)
        truncate : bool
            Whether or not to do an svd and truncate the resulting
            boundary mpo

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
    peps = peps_tools.flip_peps(peps)
    col = Nx-col

    # Loop through the columns, creating a boundary mpo for each
    bound_mpo = None
    for colind in range(col-1):
        mpiprint(4,'Updating Right boundary mpo')
        bound_mpo = update_left_bound_mpo(peps[colind][:], bound_mpo, chi=chi, singleLayer=singleLayer, truncate=truncate)

    # Return result
    return bound_mpo

