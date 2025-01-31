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

def init_left_bmpo_sl(bra, ket=None, chi=4, truncate=True,allow_normalize=False):
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
        #tmpprint('\t\t\t\tAdding row: {}'.format(row))
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
        #tmpprint('\t\t\t\tDoing MPS apply svd')
        bound_mps = bound_mps.apply_svd(chi)
        if DEBUG:
            mpiprint(6,'Computing resulting bmpo norm')
            norm1 = bound_mps.norm()
            mpiprint(0,'Init BMPO Canonicalization Norm Difference for chi={}: {} ({},{})'.format(chi,abs(norm0-norm1)/abs(norm0),norm0,norm1))
    if allow_normalize:
        bound_mps[0] /= bound_mps[0].abs().max()**(1./2.)

    return bound_mps

def left_bmpo_sl_add_ket(ket,bound_mpo,Ny,chi=4,truncate=True,allow_normalize=False):
    """
    Add the ket layer to the boundary mpo
    """
    mpiprint(4,'Adding Ket')

    # Make list to hold resulting mpo
    bound_mpo_new = []

    for row in range(Ny):
        #tmpprint('     Adding row: {}'.format(row))
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
        #tmpprint('     Doing MPS apply svd')
        bound_mps = bound_mps.apply_svd(chi)
        if DEBUG:
            mpiprint(6,'Computing resulting bmpo norm')
            norm1 = bound_mps.norm()
            mpiprint(0,'Add ket BMPO Canonicalization Norm Difference for chi={}: {} ({},{})'.format(chi,abs(norm0-norm1)/abs(norm0),norm0,norm1))
    if allow_normalize:
        bound_mps[0] /= bound_mps[0].abs().max()**(1./2.)

    return bound_mps

def left_bmpo_sl_add_bra(bra,bound_mpo,Ny,chi=4,truncate=True,allow_normalize=False):
    """
    Add the bra layer to the boundary mpo
    """
    mpiprint(4,'Adding Bra')
    # Make list to hold resulting mpo
    bound_mpo_new = []

    for row in range(Ny):
        #tmpprint('     Adding row: {}'.format(row))

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
        #tmpprint('     Doing MPS apply svd')
        bound_mps = bound_mps.apply_svd(chi)
        if DEBUG:
            mpiprint(6,'Computing resulting bmpo norm')
            norm1 = bound_mps.norm()
            mpiprint(0,'Add bra BMPO Canonicalization Norm Difference for chi={}: {} ({},{})'.format(chi,abs(norm0-norm1)/abs(norm0),norm0,norm1))
    if allow_normalize:
        bound_mps[0] /= bound_mps[0].abs().max()**(1./2.)

    return bound_mps

def left_bmpo_sl(bra, bound_mpo, chi=4,truncate=True,ket=None,allow_normalize=False):
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
    #tmpprint('    Adding ket')
    bound_mpo = left_bmpo_sl_add_ket(ket,bound_mpo,Ny,chi=chi,truncate=truncate,allow_normalize=allow_normalize)
    # Second Layer (bra) ####################################
    #tmpprint('    Adding bra')
    bound_mpo = left_bmpo_sl_add_bra(bra,bound_mpo,Ny,chi=chi,truncate=truncate,allow_normalize=allow_normalize)

    # Return result
    return bound_mpo

#@profile
def left_update_sl(peps_col, bound_mpo, chi=4,truncate=True,ket=None,allow_normalize=False):
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
        #tmpprint('   Initial BMPO')
        bound_mpo = init_left_bmpo_sl(peps_col,chi=chi,truncate=truncate,ket=ket,allow_normalize=allow_normalize)
    # Otherwise update is generic
    else:
        # Start from bottom of the column
        #tmpprint('   Adding BMPO')
        bound_mpo = left_bmpo_sl(peps_col,bound_mpo,chi=chi,truncate=truncate,ket=ket,allow_normalize=allow_normalize)
    return bound_mpo

def left_update(peps_col,bound_mpo,chi=4,ket=None):
    mpiprint(0,'Only single layer environment implemented')
    raise NotImplemented

def update_left_bound_mpo(peps_col, bound_mpo, chi=4, singleLayer=True,truncate=True,ket_col=None,allow_normalize=False):
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
        return left_update_sl(peps_col,bound_mpo,chi=chi,truncate=truncate,ket=ket_col,allow_normalize=allow_normalize)
    else:
        return left_update(peps_col,bound_mpo,chi=chi,truncate=truncate,ket=ket_col)

def calc_left_bound_mpo(peps,col,chi=4,singleLayer=True,truncate=True,return_all=False,ket=None,allow_normalize=False,in_mem=True):
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
        in_mem: bool
            if True, then the peps tensors will all be loaded into memory
            and all calculations will be done with them in memory. If False,
            then the peps tensors will all be written to disk, then loaded as 
            needed. All bmpo tensors will be written to disk. Default
            is True

    returns:
        bound_mpo : list
            An mpo stored as a list, corresponding to the
            resulting boundary mpo.

    """
    mpiprint(2,'Computing Left boundary MPO')

    # Ensure peps are in or out of memory
    if in_mem:
        peps.from_disk()
    else:
        peps.to_disk()

    # Determine the dimensions of the peps
    Nx = len(peps)
    Ny = len(peps[0])

    # Set up initial list to store boundary mpos
    bound_mpo = [None]*(col-1)

    # Loop through the columns, creating a boundary mpo for each
    for colind in range(col-1):

        #tmpprint('  Doing Column {}'.format(colind))
        mpiprint(4,'Updating left boundary mpo')

        # Load appropriate peps/ket column
        if not in_mem:
            peps.col_from_disk(colind)
            if ket is not None:
                ket.col_from_disk(colind)
        
        # Specify ket column (if not None)
        if ket is not None:
            ket_col = ket[colind][:]
        else: ket_col = None

        # Update the bmpo
        if colind == 0:

            # Update for the initial column (use None as previous boundary mpo)
            bound_mpo[colind] = update_left_bound_mpo(peps[colind][:], 
                                                      None, 
                                                      chi=chi, 
                                                      singleLayer=singleLayer,
                                                      truncate=truncate,
                                                      ket_col=ket_col,
                                                      allow_normalize=allow_normalize)

        else:

            # Update for remaining columns
            bound_mpo[colind] = update_left_bound_mpo(peps[colind][:], 
                                                      bound_mpo[colind-1], 
                                                      chi=chi, 
                                                      singleLayer=singleLayer,
                                                      truncate=truncate,
                                                      ket_col=ket_col,
                                                      allow_normalize=allow_normalize)

            # Write previous bound_mpo to disk (if not in_mem)
            if not in_mem:
                bound_mpo[colind-1].to_disk()

        # Write the peps/ket column to disk
        if not in_mem:
            peps.col_to_disk(colind)
            if ket is not None:
                ket.col_to_disk(colind)

    # Write final bmpo to disk
    if not in_mem:
        bound_mpo[-1].to_disk()

    # Return result
    if return_all:
        return bound_mpo
    else:
        return bound_mpo[-1]

#@profile
def calc_right_bound_mpo(peps,col,chi=4,singleLayer=True,truncate=True,return_all=False,ket=None,allow_normalize=False,in_mem=True):
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
        in_mem: bool
            if True, then the peps tensors will all be loaded into memory
            and all calculations will be done with them in memory. If False,
            then the peps tensors will all be written to disk, then loaded as 
            needed. All bmpo tensors will be written to disk. Default
            is True

    returns:
        bound_mpo : list
            An mpo stored as a list, corresponding to the
            resulting boundary mpo.

    """
    mpiprint(2,'Computing Left boundary MPO')

    # Ensure peps are in or out of memory
    if in_mem:
        peps.from_disk()
    else:
        peps.to_disk()

    # Determine the dimensions of the peps
    Nx = len(peps)
    Ny = len(peps[0])

    # Flip the peps
    peps.flip()
    #peps = flip_peps(peps)
    if ket is not None:
        ket.flip()
        #ket = flip_peps(ket)
    col = Nx-col
   
    # Set up initial list to store boundary mpos
    bound_mpo = [None]*(col-1)

    # Loop through the columns, creating a boundary mpo for each
    for colind in range(col-1):

        #tmpprint('  Doing Column {}'.format(colind))
        mpiprint(4,'Updating boundary mpo')

        # Load appropriate peps column
        if not in_mem:
            peps.col_from_disk(colind)
            if ket is not None:
                ket.col_from_disk(colind)
        
        # Specify ket column (if not None)
        if ket is not None:
            ket_col = ket[colind][:]
        else: ket_col = None

        # Update the boundary MPO
        if colind == 0:

            # Update for the initial column (use None as previous boundary mpo)
            bound_mpo[colind] = update_left_bound_mpo(peps[colind][:], 
                                                      None, 
                                                      chi=chi, 
                                                      singleLayer=singleLayer, 
                                                      truncate=truncate, 
                                                      ket_col=ket_col,
                                                      allow_normalize=allow_normalize)

        else:

            # Update for remaining columns
            bound_mpo[colind] = update_left_bound_mpo(peps[colind][:], 
                                                      bound_mpo[colind-1], 
                                                      chi=chi, 
                                                      singleLayer=singleLayer, 
                                                      truncate=truncate, 
                                                      ket_col=ket_col,
                                                      allow_normalize=allow_normalize)

            # Write previous bound_mpo to disk (if not in_mem)
            if not in_mem:
                bound_mpo[colind-1].to_disk()

        # Write the peps/ket column to disk
        if not in_mem:
            peps.col_to_disk(colind)
            if ket is not None:
                ket.col_to_disk(colind)

    # Write final bmpo to disk
    if not in_mem:
        bound_mpo[-1].to_disk()

    # Unflip the peps
    peps.flip()
    #peps = flip_peps(peps)
    if ket is not None:
        ket.flip()
        #ket = flip_peps(ket)

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

            # Rotate clockwise
            if clockwise:

                # Copy Correct Tensor
                rpeps[y][Nx-1-x] = peps[x][y].copy()

                # Load tensor if not initially in memory
                init_in_mem = rpeps[y][Nx-1-x].in_mem
                if not init_in_mem:
                    rpeps[y][Nx-1-x].from_disk()

                # Do the rotation
                rpeps[y][Nx-1-x] = rpeps[y][Nx-1-x].transpose([1,3,2,4,0])

                # Write tensor back to disk if initially not in memory
                if not init_in_mem:
                    rpeps[y][Nx-1-x].to_disk()

            # Rotate counter clockwise
            else:

                # Copy Correct Tensor
                rpeps[Ny-1-y][x] = peps[x][y].copy()

                # Load tensor if not initially in memory
                init_in_mem = rpeps[Ny-1-y][x].in_mem
                if not init_in_mem:
                    rpeps[Ny-1-y][x].from_disk()

                # Reorder Indices to do rotation
                rpeps[Ny-1-y][x] = rpeps[Ny-1-y][x].transpose([4,0,2,1,3])

                # Write tensor back to disk if initially not in memory
                if not init_in_mem:
                    rpeps[Ny-1-y][x].to_disk()

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

            # Load tensors for transpose (if out of memory)
            init_in_mem = fpeps[x][y].in_mem
            if not init_in_mem:
                fpeps[x][y].from_disk()

            # Transpose to reorder indices
            fpeps[x][y] = fpeps[x][y].transpose([3,1,2,0,4])

            # Write tensors back to disk (if originally out of memory)
            if not init_in_mem:
                fpeps[x][y].to_disk()

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
    # Ensure all peps column elements are in memory
    for i in range(len(peps_col)):
        if not peps_col[i].in_mem:
            raise ValueError('PEPS column tensor {} not in memory for calc_peps_col_norm'.format(i))

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
    # Ensure all peps column elements are in memory
    for i in range(len(peps_col)):
        if not peps_col[i].in_mem:
            raise ValueError('PEPS column tensor {} not in memory for calc_peps_col_norm'.format(i))

    # Convert peps column to an mps by lumping indices
    mps = peps_col_to_mps(peps_col)

    # Compute the norm of that mps
    norm = 0.5*mps.norm()

    # Return the resulting norm
    return norm

def thermal_peps_tensor(Nx,Ny,x,y,d,D,Zn=None,dZn=None,backend='numpy',dtype=float_,in_mem=True):
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
        in_mem : bool
            Whether the peps tensors should be stored in memory or on disk

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
        Dl = int(Dl/Znl)
        Dr = int(Dr/Znr)
        Dd = int(Dd/Znd)
        Du = int(Du/Znu)
        d  = int(d/dZn)

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

    # Store on disk (if wanted)
    if not in_mem: ten.to_disk()

    # Return result
    return ten

def rand_peps_tensor(Nx,Ny,x,y,d,D,Zn=None,dZn=None,backend='numpy',dtype=float_,in_mem=True):
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
        in_mem : bool
            Whether the PEPS tensor should be stored in memory or on disk.
            Default is True (i.e. in memory)

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
        Dl = int(Dl/Znl)
        Dr = int(Dr/Znr)
        Dd = int(Dd/Znd)
        Du = int(Du/Znu)
        d  = int(d/dZn)

        # Create sym argument
        sym = ['+++--',
               [range(Znl),range(Znd),range(dZn),range(Znr),range(Znu)],
               0,
               Zn]

    # Create the random tensor
    dims = (Dl,Dd,d,Dr,Du)
    ten = rand(dims,sym,backend=backend,dtype=dtype)
    #ten = 0.95*ones(dims,sym,backend=backend,dtype=dtype) + 0.1*rand(dims,sym,backend=backend,dtype=dtype)
    #ten = 0.9995*ones(dims,sym,backend=backend,dtype=dtype) + 0.001*rand(dims,sym,backend=backend,dtype=dtype)

    # Push to disk (if wanted)
    if not in_mem:
        ten.to_disk()
    
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
    # Ensure all peps column elements are in memory
    for i in range(len(peps_col)):
        if not peps_col[i].in_mem:
            raise ValueError('PEPS column tensor {} not in memory for normalize peps column'.format(i))

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
    if not np.isfinite(const):
        raise ValueError('Multiplying PEPS by {} is not valid'.format(const))
    Nx = len(peps)
    Ny = len(peps[0])
    for xind in range(Nx):
        for yind in range(Ny):
            peps[xind][yind] *= const
    return peps

def normalize_peps(peps,max_iter=100,norm_tol=1e-2,exact_norm_tol=3,chi=10,up=5.0,
                    down=0.0,singleLayer=True,ket=None,pf=False,in_mem=True):
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
        exact_norm_tol : float
            We require the measured norm to be within the bounds
            10^(-norm_tol) < norm < 10^(norm_tol) before we do
            exact arithmetic to get the norm very close to 1. Default
            is 1.
        norm_tol : int
            How close the norm must be to 1. to consider the norm 
            to be sufficiently well converged
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
        ket : peps object
            If you would like the ket to be 'normalized', such that 
            when contracted with another peps, the contraction is equal
            to one. Only the peps (not ket) will be altered to attempt
            the normalization
        pf: bool
            If True, then we will normalize as though this is a partition
            function instead of a contraction between to peps
        in_mem: bool
            if True, then the peps tensors will all be loaded into memory
            and all calculations will be done with them in memory, default
            is True

    Returns:
        norm : float
            The approximate norm of the PEPS after the normalization
            procedure
        peps : list
            The normalized version of the PEPS, given as a PEPS object

    """
    # Make sure PEPS tensors are in or out of mem
    #tmpprint(' Normalizing PEPS')
    if in_mem:
        peps.from_disk()
    else:
        peps.to_disk()

    # Figure out peps size
    Nx = peps.Nx
    Ny = peps.Ny
    be = peps[0][0].backend

    # Power changes if partition function or norm
    if pf: pwr = -1./(Nx*Ny)
    else: pwr = -1./(2*Nx*Ny)

    # Make sure PEPS entries are not really huge or miniscule
    maxval = peps.max_entry()
    if (maxval > 10**4) or (maxval < 10**-4):
        peps = multiply_peps_elements(peps.copy(),2/maxval)
    if ket is not None:
        maxval = peps.max_entry()
        if (maxval > 10**-4) or (maxval < 10**-4):
            peps = multiply_peps_elements(peps.copy(),2/maxval)

    # Check if state is already easily normalized
    try:
        z = calc_peps_norm(peps,chi=chi,singleLayer=singleLayer,ket=ket,in_mem=in_mem)
    except Exception as e:
        z = None
    #tmpprint('  Initial Norm = {}'.format(z))
    if (z is None) or (not (z < 10.**(-1*norm_tol) or z > 10.**(norm_tol))):
        if z is not None:
            sfac = z**pwr
            peps_try = multiply_peps_elements(peps.copy(),sfac)
            z = calc_peps_norm(peps_try,chi=chi,singleLayer=singleLayer,ket=ket,in_mem=in_mem)
            if abs(z-1.) < norm_tol: 
                return z, peps_try
        else:
            z = None
            peps_try = peps.copy()

    # Begin search --------
    niter  = 0
    scale = (up+down)/2.
    z     = None
    converged = False

    while not converged:

        # Update Iteration count
        niter += 1

        # Calculate norm
        peps_try = multiply_peps_elements(peps.copy(),scale)
        zprev = z
        z = None
        try:
            z = calc_peps_norm(peps_try,chi=chi,singleLayer=singleLayer,ket=ket,in_mem=in_mem)
            z = abs(z)
        except Exception as e:
            pass
        #print('scale: {}, up: {}, down: {}, norm: {}'.format(scale,up,down,z))

        # Determine next scale (for infinite or failed norm result)
        if (z == None) or (not np.isfinite(z)):

            # Replace None with nan (so can be compared using '<')
            if z == None: z = np.nan

            # Set up -> scale (if previous norm was infinite/None/nan/<1)
            if ((zprev == None) or (not np.isfinite(zprev))) or (zprev < 1.):
                up = scale if (scale is not None) else up
                scale = (up+down)/2.
            # Set down -> scale (if previous norm was > 1)
            else:
                down = scale if (scale is not None) else down
                scale = (up+down)/2.

        # adjust scale to make norm in target region
        else:

            # Check if sufficiently well converged
            if abs(z-1.0) < norm_tol:
                mpiprint(2, 'converged scale = {}, norm = {}'.format(scale,z))
                converged = True

            # Check if we are still far away from convergence
            elif z < 10.0**(-1*norm_tol) or z > 10.0**(norm_tol) or be.isnan(z):
                if z > 1.0 or be.isnan(z):
                    up = scale if (scale is not None) else up
                    scale = (up+down)/2.0
                else:
                    down = scale if (scale is not None) else down
                    scale = (up+down)/2.0

            # Close to convergence, apply "exact" scale
            else:
                sfac = z**pwr
                scale = sfac*scale if (scale is not None) else sfac
                mpiprint(2, 'apply exact scale: {}'.format(scale))

        # Print Results of current step
        mpiprint(2, 'step={}, (down,up)=({},{}), scale={}, norm={}'.format(
                                                        niter,down,up,scale,z))

        # Check if we have exceeded the maximum number of iterations
        if niter == max_iter:
            mpiprint(4, 'binarySearch normalization exceeds max_iter... terminating')
            converged=True

    # Return normalized PEPS and norm 
    return z, peps_try

def calc_peps_norm(_peps,chi=4,singleLayer=True,ket=None,allow_normalize=False,in_mem=True):
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
        in_mem: bool
            if True, then the peps tensors will all be loaded into memory
            and all calculations will be done with them in memory, default
            is True

    Returns:
        norm : float
            The (approximate) norm of the PEPS
    """
    #tmpprint('Calculating PEPS norm')

    # Absorb Lambda tensors if needed
    if _peps.ltensors is not None:
        peps = _peps.copy()
        peps.absorb_lambdas()
    else:
        #tmpprint(' Copying PEPS')
        peps = _peps.copy()
    if ket is not None and ket.ltensors is not None:
        ket = ket.copy()
        ket.absorb_lambdas()
    elif ket is not None:
        ket = ket.copy()

    # Load tensors or send to memory
    if in_mem:
        peps.from_disk()
    else:
        peps.to_disk()

    # Get PEPS Dims
    Nx = len(peps)
    Ny = len(peps[0])

    # Get the boundary MPO from the left (for the furthest right column)
    #tmpprint(' Calculating lbmpo')
    left_bound_mpo  = calc_left_bound_mpo(peps,Nx,chi=chi,singleLayer=singleLayer,ket=ket,allow_normalize=allow_normalize,in_mem=in_mem)

    # Get the boundary MPO from the right (for the furthest right column)
    #tmpprint(' Calculating rbmpo')
    right_bound_mpo = calc_right_bound_mpo(peps,Nx-2,chi=chi,singleLayer=singleLayer,ket=ket,allow_normalize=allow_normalize,in_mem=in_mem)

    # Load needed bmpos
    if not in_mem:
        left_bound_mpo.from_disk()
        right_bound_mpo.from_disk()

    # Contract the two MPOs
    #tmpprint(' Contracting two bmpos')
    norm = left_bound_mpo.contract(right_bound_mpo)

    # Return result
    return abs(norm)

def make_thermal_peps(Nx,Ny,d,D,Zn=None,dZn=None,backend='numpy',dtype=float_,in_mem=True):
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
        in_mem : bool
            Whether the peps tensors should be stored in memory or on disk

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

    # Place thermal tensors into the PEPS
    for x in range(Nx):
        for y in range(Ny):
            tensors[x][y] = thermal_peps_tensor(Nx,Ny,x,y,d,D,
                                                Zn=Zn,
                                                dZn=dZn,
                                                backend=backend,
                                                dtype=dtype,
                                                in_mem=in_mem)

    return tensors

def make_rand_peps(Nx,Ny,d,D,Zn=None,dZn=None,backend='numpy',dtype=float_,in_mem=True):
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
        in_mem : bool
            Whether the peps tensors should be stored in memory or on disk.
            Default is True

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
            tensors[x][y] = rand_peps_tensor(Nx,Ny,x,y,d,D,
                                             Zn=Zn,
                                             dZn=dZn,
                                             backend=backend,
                                             dtype=dtype,
                                             in_mem=True)

        # At the end of each column, make the norm smaller
        tensors[x][:] = normalize_peps_col(tensors[x][:])

        # And write to disk
        if not in_mem:
            for y in range(Ny):
                tensors[x][y].to_disk()

    return tensors

def thermal_lambda_tensor(D,Zn=None,backend='numpy',dtype=float_,in_mem=True):
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
        in_mem : bool
            If True, then the tensor will be stored in local memory. Otherwise,
            it will be written to disk

    Returns:
        ten : ndarray
            A random tensor with the correct dimensions
            for the given site
    """
    # Determine symmetry
    sym = None
    if Zn is not None:
        sym = ['+-',[range(Zn)]*2,0,Zn]
        D = int(D/Zn)

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

    # Write to disk if needed
    if not in_mem:
        l.to_disk()

    # Return result
    return l

def rand_lambda_tensor(D,Zn=None,backend='numpy',dtype=float_,in_mem=True):
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
        in_mem : bool
            If True (default), then the tensor will be stored in local
            memory. Otherwise it will be written to disk

    Returns:
        ten : ndarray
            A random tensor with the correct dimensions
            for the given site
    """
    # Determine symmetry
    sym = None
    if Zn is not None:
        sym = ['+-',[range(Zn)]*2,0,Zn]
        D = int(D/Zn)

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

    # Write to disk (if wanted)
    if not in_mem:
        l.to_disk()

    # Return result
    return l

def make_thermal_lambdas(Nx,Ny,D,Zn=None,backend='numpy',dtype=float_,in_mem=True):
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
            tmp += [thermal_lambda_tensor(D,Zn=Zn,backend=backend,dtype=dtype,in_mem=in_mem)]
        vert += [tmp]

    # Lambda tensors along horizontal bonds
    horz = []
    for x in range(Nx-1):
        tmp = []
        for x in range(Ny):
            tmp += [thermal_lambda_tensor(D,Zn=Zn,backend=backend,dtype=dtype,in_mem=in_mem)]
        horz += [tmp]

    # Add horizontal and vertical lambdas to tensor list
    tensors = [vert,horz]
    return tensors

def make_rand_lambdas(Nx,Ny,D,Zn=None,backend='numpy',dtype=float_,in_mem=True):
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
            tmp += [rand_lambda_tensor(D,Zn=Zn,backend=backend,dtype=dtype,in_mem=in_mem)]
        vert += [tmp]

    # Lambda tensors along horizontal bonds
    horz = []
    for x in range(Nx-1):
        tmp = []
        for x in range(Ny):
            tmp += [rand_lambda_tensor(D,Zn=Zn,backend=backend,dtype=dtype,in_mem=in_mem)]
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
                mpiprint(0,'Init top BMPO Canonicalization Norm Difference for chi={}: {} ({},{})'.format(chi,abs(norm0-norm1)/abs(norm0),norm0,norm1))
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
                mpiprint(0,'Add ket top BMPO Canonicalization Norm Difference for chi={}: {} ({},{})'.format(chi,abs(norm0-norm1)/abs(norm0),norm0,norm1))

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
                mpiprint(0,'Add bra top BMPO Canonicalization Norm Difference for chi={}: {} ({},{})'.format(chi,abs(norm0-norm1)/abs(norm0),norm0,norm1))

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
                mpiprint(0,'Init bot BMPO Canonicalization Norm Difference for chi={}: {} ({},{})'.format(chi,abs(norm0-norm1)/abs(norm0),norm0,norm1))
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
                mpiprint(0,'Add ket bot BMPO Canonicalization Norm Difference for chi={}: {} ({},{})'.format(chi,abs(norm0-norm1)/abs(norm0),norm0,norm1))

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
                mpiprint(0,'Add bra bot BMPO Canonicalization Norm Difference for chi={}: {} ({},{})'.format(chi,abs(norm0-norm1)/abs(norm0),norm0,norm1))

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
    # Check if stuff is in memory (or needs loading)
    in_mem_bra = bra.in_mem
    in_mem_ket = ket.in_mem
    in_mem_left1 = left1.in_mem
    in_mem_left2 = left2.in_mem
    in_mem_right1 = right1.in_mem
    in_mem_right2 = right2.in_mem
    if prev_env is not None:
        in_mem_prev_env = prev_env.in_mem
    else:
        in_mem_prev_env = True

    # Load stuff that is not in memory
    if not in_mem_bra: bra.from_disk()
    if not in_mem_ket: ket.from_disk()
    if not in_mem_left1: left1.from_disk()
    if not in_mem_left2: left2.from_disk()
    if not in_mem_right1: right1.from_disk()
    if not in_mem_right2: right2.from_disk()
    if not in_mem_prev_env: prev_env.from_disk()

    # Compute first bottom environment
    if prev_env is None:
        tmp = einsum('ldpru,NlO->uONdpr',ket,left2).remove_empty_ind(0).remove_empty_ind(0)
        tmp = einsum('Ndpr,nro->oNdpn',tmp,right2).remove_empty_ind(0)
        tmp = einsum('Ndpn,LDpRU->UNdLDRn',tmp,bra).remove_empty_ind(0)
        tmp = einsum('NdLDRn,MLN->MdDRn',tmp,left1)
        top_env = einsum('MdDRn,mRn->MdDm',tmp,right1)

    # Add on to top env
    else:
        tmp = einsum('ldpru,OuUo->OldprUo',ket,prev_env)
        tmp = einsum('OldprUo,NlO->NdprUo',tmp,left2)
        tmp = einsum('NdprUo,nro->NdpUn',tmp,right2)
        tmp = einsum('NdpUn,LDpRU->NdLDRn',tmp,bra)
        tmp = einsum('NdLDRn,MLN->MdDRn',tmp,left1)
        top_env = einsum('MdDRn,mRn->MdDm',tmp,right1)

    # Cache stuff that is not in memory
    if not in_mem_bra: bra.to_disk()
    if not in_mem_ket: ket.to_disk()
    if not in_mem_left1: left1.to_disk()
    if not in_mem_left2: left2.to_disk()
    if not in_mem_right1: right1.to_disk()
    if not in_mem_right2: right2.to_disk()
    if not in_mem_prev_env: prev_env.to_disk()

    # Return result
    return top_env

#@profile
def calc_top_envs(bra_col,left_bmpo,right_bmpo,ket_col=None,in_mem=True):
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

    # Compute top environment
    top_env = [None]*Ny
    for row in reversed(range(Ny)):
        
        # Get previous Environemnt
        if row == Ny-1: prev_env = None
        else: prev_env = top_env[row+1]

        # Make sure everything we need is loaded
        if not in_mem:
            bra_col[row].from_disk()
            ket_col[row].from_disk()
            left_bmpo[2*row].from_disk()
            left_bmpo[2*row+1].from_disk()
            right_bmpo[2*row].from_disk()
            right_bmpo[2*row+1].from_disk()
            if prev_env is not None:
                prev_env.from_disk()

        # Update the top environments
        top_env[row] = update_top_env(bra_col[row],
                                      ket_col[row],
                                      left_bmpo[2*row],
                                      left_bmpo[2*row+1],
                                      right_bmpo[2*row],
                                      right_bmpo[2*row+1],
                                      prev_env)

        # Write tensors back to disk (if needed)
        if not in_mem:
            bra_col[row].to_disk()
            ket_col[row].to_disk()
            left_bmpo[2*row].to_disk()
            left_bmpo[2*row+1].to_disk()
            right_bmpo[2*row].to_disk()
            right_bmpo[2*row+1].to_disk()
            if row != Ny-1:
                top_env[row+1].to_disk()

    # Write final top env to disk
    if not in_mem:
        top_env[row].to_disk()

    # Return Result
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
    # Check if stuff is in memory (or needs loading)
    in_mem_bra = bra.in_mem
    in_mem_ket = ket.in_mem
    in_mem_left1 = left1.in_mem
    in_mem_left2 = left2.in_mem
    in_mem_right1 = right1.in_mem
    in_mem_right2 = right2.in_mem
    if prev_env is not None:
        in_mem_prev_env = prev_env.in_mem
    else:
        in_mem_prev_env = True

    # Load stuff that is not in memory
    if not in_mem_bra: bra.from_disk()
    if not in_mem_ket: ket.from_disk()
    if not in_mem_left1: left1.from_disk()
    if not in_mem_left2: left2.from_disk()
    if not in_mem_right1: right1.from_disk()
    if not in_mem_right2: right2.from_disk()
    if not in_mem_prev_env: prev_env.from_disk()

    # Compute first bottom environment
    if prev_env is None:
        tmp = einsum('LDPRU,MLN->DMNPUR',bra,left1).remove_empty_ind(0).remove_empty_ind(0)
        tmp = einsum('NPUR,mRn->mNPUn',tmp,right1).remove_empty_ind(0)
        tmp = einsum('NPUn,ldPru->dNlurUn',tmp,ket).remove_empty_ind(0)
        tmp = einsum('NlurUn,NlO->OurUn',tmp,left2)
        bot_env = einsum('OurUn,nro->OuUo',tmp,right2)

    # Update bottom environemnt
    else:
        tmp = einsum('LDPRU,MdDm->MdLPURm',bra,prev_env)
        tmp = einsum('MdLPURm,MLN->NdPURm',tmp,left1)
        tmp = einsum('NdPURm,mRn->NdPUn',tmp,right1)
        tmp = einsum('NdPUn,ldPru->NlurUn',tmp,ket)
        tmp = einsum('NlurUn,NlO->OurUn',tmp,left2)
        bot_env = einsum('OurUn,nro->OuUo',tmp,right2)

    # Cache stuff that is not in memory
    if not in_mem_bra: bra.to_disk()
    if not in_mem_ket: ket.to_disk()
    if not in_mem_left1: left1.to_disk()
    if not in_mem_left2: left2.to_disk()
    if not in_mem_right1: right1.to_disk()
    if not in_mem_right2: right2.to_disk()
    if not in_mem_prev_env: prev_env.to_disk()

    # Return result
    return bot_env

#@profile
def calc_bot_envs(bra_col,left_bmpo,right_bmpo,ket_col=None,in_mem=True):
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

        # Get previous environment
        if row == 0: prev_env = None
        else: prev_env = bot_env[row-1]

        # Make sure everything we need is loaded
        if not in_mem:
            bra_col[row].from_disk()
            ket_col[row].from_disk()
            left_bmpo[2*row].from_disk()
            left_bmpo[2*row+1].from_disk()
            right_bmpo[2*row].from_disk()
            right_bmpo[2*row+1].from_disk()
            if prev_env is not None:
                prev_env.from_disk()

        # Update the top environments
        bot_env[row] = update_bot_env(bra_col[row],
                                      ket_col[row],
                                      left_bmpo[2*row],
                                      left_bmpo[2*row+1],
                                      right_bmpo[2*row],
                                      right_bmpo[2*row+1],
                                      prev_env)

        # Write tensors back to disk (if needed)
        if not in_mem:
            bra_col[row].to_disk()
            ket_col[row].to_disk()
            left_bmpo[2*row].to_disk()
            left_bmpo[2*row+1].to_disk()
            right_bmpo[2*row].to_disk()
            right_bmpo[2*row+1].to_disk()
            if row-1 > 0:
                bot_env[row-1].to_disk()

    # Write final top env to disk
    if not in_mem:
        bot_env[row].to_disk()

    # Return result
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

#@profile
def make_N_positive(N,hermitian=True,positive=True,reduced=True):
    """
    """
    hermitian,positive=False,False
    # Get a hermitian approximation of the environment
    if hermitian:
        if reduced:
            N1 = N.copy()
            N1 = N1.transpose([0,2,1,3])
            N = N.transpose([1,3,0,2])
            N = (N+N1)/2.
            N1 = N.copy()
            N = einsum('UDab,abud->UuDd',N,N1)
        else:
            N1 = N.copy()
            N1 = N1.transpose([0,2,4,6,8,10,1,3,5,7,9,11])
            N = N.transpose([1,3,5,7,9,11,0,2,4,6,8,10])
            N = (N+N1)/2.
            N1 = N.copy()
            N = einsum('ldrkustvwxyz,tvwxyzLDRKUS->lLdDrRkKuUsS',N,N1)

    # Get a positive approximation of the environment
    if positive:
        try:
            if reduced:
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
            else:
                if N.sym is None:
                    N = N.transpose([0,2,4,6,8,10,1,3,5,7,9,11])
                    n0 = np.prod([N.ten.shape[i] for i in N.legs[0]])
                    n1 = np.prod([N.ten.shape[i] for i in N.legs[1]])
                    n2 = np.prod([N.ten.shape[i] for i in N.legs[2]])
                    n3 = np.prod([N.ten.shape[i] for i in N.legs[3]])
                    n4 = np.prod([N.ten.shape[i] for i in N.legs[4]])
                    n5 = np.prod([N.ten.shape[i] for i in N.legs[5]])
                    n6 = np.prod([N.ten.shape[i] for i in N.legs[6]])
                    n7 = np.prod([N.ten.shape[i] for i in N.legs[7]])
                    n8 = np.prod([N.ten.shape[i] for i in N.legs[8]])
                    n9 = np.prod([N.ten.shape[i] for i in N.legs[9]])
                    n10 = np.prod([N.ten.shape[i] for i in N.legs[10]])
                    n11 = np.prod([N.ten.shape[i] for i in N.legs[11]])
                    Nmat = N.backend.reshape(N.ten,(n0*n1*n2*n3*n4*n5,n6*n7*n8*n9*n10*n11))
                    u,v = N.backend.eigh(Nmat)
                    u = pos_sqrt_vec(u)
                    Nmat = N.backend.einsum('ij,j,kj->ik',v,u,v)
                    N.ten = Nmat.reshape(N.shape)
                    N = N.transpose([0,6,1,7,2,8,3,9,4,10,5,11])
                else:
                    N = N.copy().transpose([0,2,4,6,8,10,1,3,5,7,9,11])
                    Nmat = N.ten.make_sparse()
                    (N0,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,N11,n0,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11) = Nmat.shape
                    Nmat = Nmat.transpose([0,12,1,13,2,14,3,15,4,16,5,17,6,18,7,19,8,20,9,21,10,22,11,23])
                    Nmat = Nmat.reshape((N0*n0*N1*n1*N2*n2*N3*n3*N4*n4*N5*n5,N6*n6*N7*n7*N8*n8*N9*n9*N10*n10*N11*n11))
                    u,v = N.backend.eigh(Nmat)
                    u = pos_sqrt_vec(u)
                    Nmat = N.backend.einsum('ij,j,kj->ik',v,u,v)
                    Nmat = Nmat.reshape((N0,n0,N1,n1,N2,n2,N3,n3,N4,n4,N5,n5,N6,n6,N7,n7,N8,n8,N9,n9,N10,n10,N11,n11))
                    Nmat = Nmat.transpose([0,2,4,6,8,10,12,14,16,18,20,22,1,3,5,7,9,11,13,15,17,19,21,23])
                    delta = N.ten.get_irrep_map()
                    Nmat = N.backend.einsum('ABCDEFGHIJKLabcdefghijkl,ABCDEFGHIJKL->ABCDEFGHIJKabcdefghijkl',Nmat,delta)
                    N.ten.array = Nmat
                    N = N.transpose([0,6,1,7,2,8,3,9,4,10,5,11])
        except Exception as e:
            mpiprint(0,'Failed to make N positive:\n\t{}'.format(e))

    return N

#@profile
def calc_local_env(bra1,bra2,ket1,ket2,env_top,env_bot,lbmpo,rbmpo,
                   reduced=True,hermitian=True,positive=True,in_mem=True):
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
        in_mem : bool
            Whether the tensors input to this function are in 
            memory. If not, tensors should be loaded first (and rewritten
            to disk afterwards). The output of this funciton, i.e.
            the local env, will always be in memory.

    """

    # Load tensors (as needed)
    if not in_mem:
        bra1.from_disk()
        bra2.from_disk()
        ket1.from_disk()
        ket2.from_disk()
        if env_top is not None: env_top.from_disk()
        if env_bot is not None: env_bot.from_disk()
        for i in range(len(lbmpo)):
            lbmpo[i].from_disk()
        for i in range(len(rbmpo)):
            rbmpo[i].from_disk()

    if reduced:
        # Get reduced tensors
        peps_b,phys_b,phys_t,peps_t = reduce_tensors(bra1,bra2)
        ket_b,phys_bk,phys_tk,ket_t = reduce_tensors(ket1,ket2)

        # Compute bottom half of environment
        if env_bot is None:
            tmp = einsum('CLB,LDRU->CDBUR',lbmpo[0],peps_b).remove_empty_ind(0).remove_empty_ind(0)
            tmp = einsum('BUR,cRb->cBUb',tmp,rbmpo[0]).remove_empty_ind(0)
            tmp = einsum('BUb,BlA->AlUb',tmp,lbmpo[1])
            tmp = einsum('AlUb,ldru->dAurUb',tmp,ket_b).remove_empty_ind(0)
            envb= einsum('AurUb,bra->AuUa',tmp,rbmpo[1])
        else:
            tmp = einsum('CdDc,CLB->BLdDc',env_bot,lbmpo[0])
            tmp = einsum('BLdDc,LDRU->BdURc',tmp,peps_b)
            tmp = einsum('BdURc,cRb->BdUb',tmp,rbmpo[0])
            tmp = einsum('BdUb,BlA->AldUb',tmp,lbmpo[1])
            tmp = einsum('AldUb,ldru->AurUb',tmp,ket_b)
            envb= einsum('AurUb,bra->AuUa',tmp,rbmpo[1])

        # Compute top half of environment
        if env_top is None:
            tmp = einsum('BlC,ldru->CuBdr',lbmpo[3],ket_t).remove_empty_ind(0).remove_empty_ind(0)
            tmp = einsum('Bdr,brc->cBdb',tmp,rbmpo[3]).remove_empty_ind(0)
            tmp = einsum('Bdb,ALB->ALdb',tmp,lbmpo[2])
            tmp = einsum('ALdb,LDRU->UAdDRb',tmp,peps_t).remove_empty_ind(0)
            envt= einsum('AdDRb,aRb->AdDa',tmp,rbmpo[2])
        else:
            tmp = einsum('CuUc,BlC->BluUc',env_top,lbmpo[3])
            tmp = einsum('BluUc,ldru->BdrUc',tmp,ket_t)
            tmp = einsum('BdrUc,brc->BdUb',tmp,rbmpo[3])
            tmp = einsum('BdUb,ALB->ALdUb',tmp,lbmpo[2])
            tmp = einsum('ALdUb,LDRU->AdDRb',tmp,peps_t)
            envt= einsum('AdDRb,aRb->AdDa',tmp,rbmpo[2])

        # Compute Environment
        N = einsum('AdDa,AuUa->uUdD',envt,envb)
        N = make_N_positive(N,hermitian=hermitian,positive=positive)

        # write tensors to disk (as needed)
        if not in_mem:
            bra1.to_disk()
            bra2.to_disk()
            ket1.to_disk()
            ket2.to_disk()
            if env_top is not None: env_top.to_disk()
            if env_bot is not None: env_bot.to_disk()
            for i in range(len(lbmpo)):
                lbmpo[i].to_disk()
            for i in range(len(rbmpo)):
                rbmpo[i].to_disk()

        # Return Results
        return peps_b, phys_b, phys_t, peps_t, ket_b, phys_bk, phys_tk, ket_t, N
    else:
        # Get the PEPS tensors
        peps_b, peps_t = bra1, bra2
        ket_b, ket_t = ket1, ket2

        # Compute bottom half of environment
        if env_bot is None:
            if lbmpo[0].is_symmetric:
                # Must determine correct signs for empty tensor (a bit overly complicated)
                symtmp = einsum('CLB,BlA->CLlA',lbmpo[0],lbmpo[1])
                symtmp = einsum('LDPRU,CLlA->DPRUClA',peps_b,symtmp)
                symtmp = einsum('cRb,DPRUClA->DPUClAcb',rbmpo[0],symtmp)
                symtmp = einsum('bra,DPUClAcb->DPUClAcra',rbmpo[1],symtmp)
                symtmp = einsum('ldPru,DPUClAcra->CdDcAuUa',ket_b,symtmp)
                # Create an empty environment
                env_bot = ones((1,1,1,1),
                               sym=[symtmp.sym[0][:4],
                                    symtmp.sym[1][:4],
                                    None,
                                    None],
                               backend=lbmpo[0].
                               backend,dtype=lbmpo[0].dtype)
            else:
                # Create an empty environment
                env_bot = ones((1,1,1,1),
                               sym=None,
                               backend=lbmpo[0].backend,
                               dtype=lbmpo[0].dtype)
        # Contract bottom half of environment
        tmp  = einsum('CdDc,CLB->BLdDc',env_bot,lbmpo[0])
        tmp  = einsum('BLdDc,cRb->BLdDRb',tmp,rbmpo[0])
        tmp  = einsum('BLdDRb,BlA->AlLdDRb',tmp,lbmpo[1])
        envb = einsum('AlLdDRb,bra->AlLdDrRa',tmp,rbmpo[1])

        # Compute top half of environment
        if env_top is None:
            if lbmpo[3].is_symmetric:
                # Must determine correct signs for empty tensor (a bit overly complicated)
                symtmp = einsum('ALB,BlC->ALlC',lbmpo[2],lbmpo[3])
                symtmp = einsum('LDPRU,ALlC->DPRUAlC',peps_t,symtmp)
                symtmp = einsum('aRb,DPRUAlC->DPUAlCab',rbmpo[2],symtmp)
                symtmp = einsum('brc,DPUAlCab->DPUAlCarc',rbmpo[3],symtmp)
                symtmp = einsum('ldPru,DPUAlCarc->CuUcDaAd',ket_t,symtmp)
                # Create an empty environment
                env_top = ones((1,1,1,1),
                               sym=[symtmp.sym[0][:4],
                                    symtmp.sym[1][:4],
                                    None,
                                    None],
                               backend=lbmpo[0].backend,
                               dtype=lbmpo[0].dtype)
            else:
                # Create an empty environment
                env_top = ones((1,1,1,1),
                               sym=None,
                               backend=lbmpo[0].backend,dtype=lbmpo[0].dtype)
        tmp  = einsum('CuUc,BlC->BluUc',env_top,lbmpo[3])
        tmp  = einsum('BluUc,brc->BluUrb',tmp,rbmpo[3])
        tmp  = einsum('BluUrb,ALB->ALluUrb',tmp,lbmpo[2])
        envt = einsum('ALluUrb,aRb->AlLuUrRa',tmp,rbmpo[2])

        # Compute Environment
        N = einsum('AkKuUsSa,AlLdDrRa->lLdDrRkKuUsS',envt,envb)
        N = make_N_positive(N,
                            hermitian=hermitian,
                            positive=positive,
                            reduced=reduced)

        # write tensors to disk (as needed)
        if not in_mem:
            bra1.to_disk()
            bra2.to_disk()
            ket1.to_disk()
            ket2.to_disk()
            if env_top is not None: env_top.to_disk()
            if env_bot is not None: env_bot.to_disk()
            for i in range(len(lbmpo)):
                lbmpo[i].to_disk()
            for i in range(len(rbmpo)):
                rbmpo[i].to_disk()

        # Return Results
        return N 

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
    else:
        # (Bra is capital, ket is lower case)
        comb1 = einsum('LDPRZ,KZQSU->LDPRKQSU', phys_b_bra, phys_t_bra)
        comb1 = einsum('LDPRKQSU,lLdDrRkKuUsS->PQldrkus', comb1, N)
        comb2 = einsum('ldprz,kzqsu->ldprkqsu', phys_b_ket, phys_t_ket)
        norm = einsum('PQldrkus,ldPrkQsu->', comb1, comb2)
        if ham is not None:
            phys_inds = einsum('PQldrkus,ldprkqsu->PQpq', comb1, comb2)
            if len(phys_inds.legs[0]) == 2:
                # Thermal state
                phys_inds.unmerge_ind(3)
                phys_inds.unmerge_ind(2)
                phys_inds.unmerge_ind(1)
                phys_inds.unmerge_ind(0)
                E = einsum('PaQbpaqb,PQpq->', phys_inds, ham)
                phys_inds.merge_inds([0,1])
                phys_inds.merge_inds([1,2])
                phys_inds.merge_inds([2,3])
                phys_inds.merge_inds([3,4])
            else:
                # Normal peps
                E = einsum('PQpq,PQpq->', phys_inds, ham)
        else:
            E = norm

    # Return Result
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

def calc_N(row,bra_col,left_bmpo,right_bmpo,top_envs,bot_envs,hermitian=True,positive=True,ket_col=None,in_mem=True,reduced=True):
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
                                 positive=positive,
                                 in_mem=in_mem,
                                 reduced=reduced)
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
                                 positive=positive,
                                 in_mem=in_mem,
                                 reduced=reduced)
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
                             positive=positive,
                             in_mem=in_mem,
                             reduced=reduced)
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
                             positive=positive,
                             in_mem=in_mem,
                             reduced=reduced)
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
        if not isinstance(E,float): E = E.to_val()
        if not isinstance(norm,float): norm = norm.to_val()
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

    # Copy bra if needed ----------------------------------
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

    # Extract needed tensors -------------------------------
    # Upper and lower environments =====
    if row == 0:
        if len(bra[0]) == 2:
            # Only two sites in column, use identity at both ends
            top,bot = None,None
        else:
            # At bottom unit cell, use identity on bottom
            top=top_envs[row+2]
            bot=None
    elif row == len(bra[0])-2:
        # At top unit cell, use identity on top
        top = None
        bot = bot_envs[row-1]
    else:
        # In the bulk, no identity needed
        top = top_envs[row+2]
        bot = bot_envs[row-1]
    # PEPS tensors =====================
    cell_bra = [[bra[0][row],bra[0][row+1]],
                [bra[1][row],bra[1][row+1]]]    
    cell_ket = [[ket[0][row],ket[0][row+1]],
                [ket[1][row],ket[1][row+1]]]
    cell_lbmpo = left_bmpo[row*2,row*2+1,row*2+2,row*2+3]
    cell_rbmpo = right_bmpo[row*2,row*2+1,row*2+2,row*2+3]
    # Flip tensors where needed ========
    # Flip bra and ket tensors
    flip_bra = [[bra[1][row].copy().transpose([3,1,2,0,4]),bra[1][row+1].copy().transpose([3,1,2,0,4])],
                [bra[0][row].copy().transpose([3,1,2,0,4]),bra[0][row+1].copy().transpose([3,1,2,0,4])]]
    flip_ket = [[ket[1][row].copy().transpose([3,1,2,0,4]),ket[1][row+1].copy().transpose([3,1,2,0,4])],
                [ket[0][row].copy().transpose([3,1,2,0,4]),ket[0][row+1].copy().transpose([3,1,2,0,4])]]
    # Flip (contracted) top/bot environments
    # Always contract bot/top env to make transpose easier
    if not contracted_env:
        if top is not None:
            flip_top = einsum('ijk,klm->ijlm',top[0],top[1]).remove_empty_ind(0)
            flip_top = einsum('jlm,mno->jlno',flip_top,top[2])
            flip_top = einsum('jlno,opq->jlnpq',flip_top,top[3])
            flip_top = einsum('jlnpq,qrs->jlnprs',flip_top,top[4])
            flip_top = einsum('jlnprs,stu->jlnprtu',flip_top,top[5]).remove_empty_ind(6)
        if bot is not None:
            flip_bot = einsum('ijk,klm->ijlm',bot[0],bot[1]).remove_empty_ind(0)
            flip_bot = einsum('jlm,mno->jlno',flip_bot,bot[2])
            flip_bot = einsum('jlno,opq->jlnpq',flip_bot,bot[3])
            flip_bot = einsum('jlnpq,qrs->jlnprs',flip_bot,bot[4])
            flip_bot = einsum('jlnprs,stu->jlnprtu',flip_bot,bot[5]).remove_empty_ind(6)
    if top is not None:
        flip_top = flip_top.transpose([5,3,4,1,2,0])
    else: flip_top = None
    if bot is not None:
        flip_bot = flip_bot.transpose([5,3,4,1,2,0])
    else: flip_bot = None
    
    # Calculation energy contribution from first MPO -------
    E1 = calc_local_nn_op_lb(ops_col[row][0],
                            cell_bra,
                            cell_ket,
                            top,
                            bot,
                            cell_lbmpo,
                            cell_rbmpo,
                            normalize=normalize,
                            chi=chi,
                            contracted_env=contracted_env)
    # Calculate energy contribution from third MPO  ---------
    # (must flip horizontally so we can use the lb procedure
    E2 = calc_local_nn_op_lb(ops_col[row][1],
                             flip_bra,
                             flip_ket,
                             flip_top,
                             flip_bot,
                             cell_rbmpo,
                             cell_lbmpo,
                             normalize=normalize,
                             chi=chi,
                             contracted_env=True)
    # Calculate energy contribution from third MPO -----------
    E3 = calc_local_nn_op_ru(ops_col[row][2],
                             cell_bra,
                             cell_ket,
                             top,
                             bot,
                             cell_lbmpo,
                             cell_rbmpo,
                             normalize=normalize,
                             chi=chi,
                             contracted_env=contracted_env)
    
    # Return resulting energy --------------------------------
    E = E1+E2+E3
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

def calc_single_column_op(peps_col,left_bmpo,right_bmpo,ops_col,
                          normalize=True,ket_col=None,in_mem=True):
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

    Kwargs:
        in_mem: bool
            if True, then the peps tensors will all be loaded into memory
            and all calculations will be done with them in memory

    """

    # Calculate top and bottom environments
    top_envs = calc_top_envs(peps_col,left_bmpo,right_bmpo,ket_col=ket_col,in_mem=in_mem)
    bot_envs = calc_bot_envs(peps_col,left_bmpo,right_bmpo,ket_col=ket_col,in_mem=in_mem)

    # Set up array to hold resulting energies
    E = peps_col[0].backend.zeros(len(ops_col))

    # Loop through rows calculating local energies
    for row in range(len(ops_col)):
        
        # Calculate environment
        res = calc_N(row,peps_col,left_bmpo,right_bmpo,top_envs,bot_envs,
                     hermitian=False,
                     positive=False,
                     ket_col=ket_col,
                     in_mem=in_mem)
        _,phys_b,phys_t,_,_,phys_bk,phys_tk,_,N = res

        # Calc the local operator
        E[row] = calc_local_op(phys_b,phys_t,N,ops_col[row],normalize=normalize,phys_b_ket=phys_bk,phys_t_ket=phys_tk)

    # Return the energy
    return E

def calc_all_column_op(peps,ops,chi=10,return_sum=True,normalize=True,ket=None,allow_normalize=False,in_mem=True):
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
        in_mem: bool
            if True, then the peps tensors will all be loaded into memory
            and all calculations will be done with them in memory

    Returns:
        val : float
            The contribution of the column's interactions to
            the observable's expectation value
    """

    # Figure out peps size
    Nx = len(peps)
    Ny = len(peps[0])

    # Compute the boundary MPOs
    right_bmpo = calc_right_bound_mpo(peps, 0,
                                      chi=chi,
                                      return_all=True,
                                      ket=ket,
                                      allow_normalize=allow_normalize,
                                      in_mem=in_mem)
    left_bmpo  = calc_left_bound_mpo (peps,Nx,
                                      chi=chi,
                                      return_all=True,
                                      ket=ket,
                                      allow_normalize=allow_normalize,
                                      in_mem=in_mem)
    ident_bmpo = identity_mps(len(right_bmpo[0]),
                              dtype=peps[0][0].dtype,
                              sym=(peps[0][0].sym is not None),
                              backend=peps.backend)

    # Set up array to store energies
    E = peps.backend.zeros((len(ops),len(ops[0])),dtype=peps[0][0].dtype)

    # Loop through all columns
    for col in range(Nx):
        
        # Get the ket column (if not None)
        if ket is None:
            ket_col = None
        else: ket_col = ket[col]

        # First column (nothing on left side)
        if col == 0:
            E[col,:] = calc_single_column_op(peps[col],
                                             ident_bmpo,
                                             right_bmpo[col],
                                             ops[col],
                                             normalize=normalize,
                                             ket_col=ket_col,
                                             in_mem=in_mem)

        # Second column (nothing on right side)
        elif col == Nx-1:
            E[col,:] = calc_single_column_op(peps[col],
                                             left_bmpo[col-1],
                                             ident_bmpo,
                                             ops[col],
                                             normalize=normalize,
                                             ket_col=ket_col,
                                             in_mem=in_mem)

        # Center columns
        else:
            E[col,:] = calc_single_column_op(peps[col],
                                             left_bmpo[col-1],
                                             right_bmpo[col],
                                             ops[col],
                                             normalize=normalize,
                                             ket_col=ket_col,
                                             in_mem=in_mem)

    # Print Energies
    mpiprint(8,'Energy [:,:] = \n{}'.format(E))

    # Return results
    if return_sum:
        return E.sum()
    else:
        return E

def calc_peps_nn_op(peps,ops,chi=10,normalize=True,ket=None,contracted_env=False,allow_normalize=False):
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
    right_bmpo = calc_right_bound_mpo(peps, 0,chi=chi,return_all=True,ket=ket,allow_normalize=allow_normalize)
    left_bmpo  = calc_left_bound_mpo (peps,Nx,chi=chi,return_all=True,ket=ket,allow_normalize=allow_normalize)
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

def calc_peps_op(peps,ops,chi=10,return_sum=True,normalize=True,ket=None,in_mem=True):
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
        in_mem: bool
            if True, then the peps tensors will all be loaded into memory
            and all calculations will be done with them in memory

    Returns:
        val : float
            The resulting observable's expectation value
    """
    # "normalize" with respect to the contraction between the two peps
    peps.normalize(ket=ket,
                   pf=True,
                   in_mem=in_mem)

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
    col_energy = calc_all_column_op(peps,ops[0],
                                    chi=chi,
                                    normalize=normalize,
                                    return_sum=return_sum,
                                    ket=ket,
                                    in_mem=in_mem)

    # Calculate contribution from interactions between rows
    peps.rotate(clockwise=True)
    if ket is not None: 
        ket.rotate(clockwise=True)
    row_energy = calc_all_column_op(peps,ops[1],
                                    chi=chi,
                                    normalize=normalize,
                                    return_sum=return_sum,
                                    ket=ket,
                                    in_mem=in_mem)
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

def load_peps(fname,in_mem=True):
    """
    Load a saved PEPS into a new PEPS object

    Args:
        fname : str
            The file which holds the saved PEPS object
        in_mem : bool
            Whether all peps tensors will be stored in local
            memory or as references to tensors stored in disk

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
    exact_norm_tol = get_dataset('exact_norm_tol')
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
                exact_norm_tol=exact_norm_tol,
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

    # Write to memory (if needed)
    if not in_mem:
        peps.to_disk()
    if canonical:
        raise ValueError('Store to disk not yet implemented for canonical PEPS')

    # Return resulting PEPS
    return peps

# -----------------------------------------------------------------
# PEPS Class

class PEPS:
    """
    A class to hold and manipulate a PEPS
    """

    #@profile
    def __init__(self,Nx=10,Ny=10,d=2,D=2,
                 chi=None,chi_norm=None,chi_op=None,
                 Zn=None,thermal=False,
                 dZn=None,canonical=False,backend='numpy',
                 singleLayer=True,dtype=float_,
                 normalize=True,norm_tol=1e-3,
                 exact_norm_tol=20.,
                 max_norm_iter=100,norm_bs_upper=10.0,norm_bs_lower=0.0,
                 fname=None,fdir='./',in_mem=True):
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
            chi_norm : int
                The boundary mpo maximum bond dimension for 
                use when the norm is computed
            chi_op : int
                The boundary mpo maximum bond dimension for 
                use when operator expectation values are computed
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
            exact_norm_tol : float
                How close to 1. the norm should be before exact
                artihmetic is used in the normalization procedure.
                See documentation of normalize_peps() function
                for more details.
            norm_tol : float
                How close to 1. the norm should be when calling peps.normalize()
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
            in_mem : bool
                Whether the peps tensors should be written to disk or 
                stored in local memory

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
        if chi_norm is None: chi_norm = chi
        if chi_op is None: chi_op = chi
        self.chi_norm = chi_norm
        self.chi_op = chi_op
        self.Zn          = Zn
        self.thermal     = thermal
        if dZn is None: dZn = Zn
        self.dZn         = dZn
        self.canonical   = canonical
        self.backend     = backend
        self.backend     = load_lib(self.backend)
        self.singleLayer = singleLayer
        self.dtype       = dtype
        self.exact_norm_tol    = exact_norm_tol
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
            #tmpprint('Making Initial Random PEPS')
            self.tensors = make_rand_peps(self.Nx,
                                          self.Ny,
                                          self.d,
                                          self.D,
                                          Zn=self.Zn,
                                          dZn=self.dZn,
                                          backend=self.backend,
                                          dtype=self.dtype,
                                          in_mem=in_mem)

        # Add in lambda "singular value" matrices
        if self.canonical:
            if thermal:
                self.ltensors = make_thermal_lambdas(self.Nx,
                                                     self.Ny,
                                                     self.D,
                                                     Zn=self.Zn,
                                                     backend=self.backend,
                                                     dtype=self.dtype,
                                                     in_mem=in_mem)
            else:
                self.ltensors = make_rand_lambdas(self.Nx,
                                                  self.Ny,
                                                  self.D,
                                                  Zn=self.Zn,
                                                  backend=self.backend,
                                                  dtype=self.dtype,
                                                  in_mem=in_mem)
        else:
            self.ltensors = None

        # Normalize the PEPS
        if normalize:
            #tmpprint('Normalize Initialized PEPS')
            self.normalize(in_mem=in_mem)

    def calc_bmpo_left(self,col,chi=4,singleLayer=True,truncate=True,return_all=False,ket=None,allow_normalize=False,in_mem=True):
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
            in_mem: bool
                if True, then the peps tensors will all be loaded into memory
                and all calculations will be done with them in memory, default
                is True

        returns:
            bound_mpo : list
                An mpo stored as a list, corresponding to the
                resulting boundary mpo.
        """
        # Get needed parameters
        if chi is None:
            chi = self.chi
        if singleLayer is None:
            singleLayer = self.singleLayer

        # Calc BMPO
        bmpo = calc_left_bound_mpo(self,
                                   col,
                                   chi=chi,
                                   singleLayer=singleLayer,
                                   truncate=truncate,
                                   return_all=return_all,
                                   ket=ket,
                                   allow_normalize=allow_normalize,
                                   in_mem=in_mem)
        
        # Return result
        return bmpo

    def calc_bmpo_right(self,col,chi=None,singleLayer=None,truncate=True,return_all=False,ket=None,allow_normalize=False,in_mem=True):
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
            in_mem: bool
                if True, then the peps tensors will all be loaded into memory
                and all calculations will be done with them in memory, default
                is True

        returns:
            bound_mpo : list
                An mpo stored as a list, corresponding to the
                resulting boundary mpo.

        """
        # Get needed parameters
        if chi is None:
            chi = self.chi
        if singleLayer is None:
            singleLayer = self.singleLayer

        # Do calculation
        bmpo = calc_right_bound_mpo(self,
                                    col,
                                    chi=chi,
                                    singleLayer=singleLayer,
                                    truncate=truncate,
                                    return_all=return_all,
                                    ket=ket,
                                    allow_normalize=allow_normalize,
                                    in_mem=in_mem)

        # Return bmpo
        return bmpo

    def calc_norm(self,chi=None,singleLayer=None,ket=None,in_mem=True):
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
            in_mem: bool
                if True, then the peps tensors will all be loaded into memory
                and all calculations will be done with them in memory, default
                is True

        Returns:
            norm : float
                The (approximate) norm of the PEPS
        """
        if chi is None: chi = self.chi
        if singleLayer is None: singleLayer = self.singleLayer
        return calc_peps_norm(self,chi=chi,singleLayer=singleLayer,ket=ket,in_mem=in_mem)

    def normalize(self,max_iter=None,norm_tol=None,exact_norm_tol=None,chi=None,up=None,down=None,
                    singleLayer=None,ket=None,pf=False,in_mem=True):
        """
        Normalize the full PEPS

        Args:
            self : PEPS Object
                The PEPS to be normalized

        Kwargs:
            max_iter : int
                The maximum number of iterations of the normalization
                procedure. Default is 20.
            norm_tol : float
                How close to 1. the norm should be when calling peps.normalize()
            exact_norm_tol : int
                We require the measured norm to be within the bounds
                10^(-exact_norm_tol) < norm < 10^(exact_norm_tol) before we do
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
            ket : peps object
                If you would like the ket to be 'normalized', such that 
                when contracted with another peps, the contraction is equal
                to one. Only the peps (not ket) will be altered to attempt
                the normalization
            pf: bool
                If True, then we will normalize as though this is a partition
                function instead of a contraction between to peps
        in_mem: bool
            if True, then the peps tensors will all be loaded into memory
            and all calculations will be done with them in memory

        Returns:
            norm : float
                The approximate norm of the PEPS after the normalization
                procedure
        """
        # Figure out good chi (if not given)
        if chi is None: chi = self.chi_norm
        if max_iter is None: max_iter = self.max_norm_iter
        if exact_norm_tol is None: exact_norm_tol = self.exact_norm_tol
        if norm_tol is None: norm_tol = self.norm_tol
        if up is None: up = self.norm_bs_upper
        if down is None: down = self.norm_bs_lower
        if singleLayer is None: singleLayer = self.singleLayer
        # Run the normalization procedure
        norm, normpeps = normalize_peps(self,
                                      max_iter = max_iter,
                                      exact_norm_tol = exact_norm_tol,
                                      norm_tol = norm_tol,
                                      chi = chi,
                                      up = up,
                                      down = down,
                                      singleLayer=singleLayer,
                                      ket=ket,
                                      pf=pf,
                                      in_mem=in_mem)
        # Copy the resulting tensors
        self.tensors = copy_peps_tensors(normpeps)
        if self.ltensors is not None:
            self.ltensors = copy_lambda_tensors(normpeps)

        return norm

    def calc_op(self,ops,chi=None,normalize=True,return_sum=True,ket=None,nn=False,contracted_env=False,in_mem=True):
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
            in_mem: bool
                if True, then the peps tensors will all be loaded into memory
                and all calculations will be done with them in memory

        Returns:
            val : float
                The resulting observable's expectation value
        """
        if chi is None: chi = self.chi_op
        # Calculate the operator's value
        if nn:
            if not in_mem: raise ValueError('Unable to do next nearest neighbor calcs with peps not in memory')
            return calc_peps_nn_op(self,ops,chi=chi,normalize=normalize,ket=ket,contracted_env=contracted_env)
        else:
            return calc_peps_op(self,ops,chi=chi,normalize=normalize,return_sum=return_sum,ket=ket,in_mem=in_mem)

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
                         exact_norm_tol=self.exact_norm_tol,
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
                     exact_norm_tol= self.exact_norm_tol,
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

    def save(self,fname=None,fdir=None):
        """
        Save the PEPS tensors
        """
        if fdir is None:
            fdir = self.fdir
        if not (fdir[-1] == '/'):
            fdir = fdir + '/'
        if fname is None:
            fname = self.fname
        if self.Zn is None:
            # Create dict to hold everything being saved
            #save_dict = dict()
            ## Add PEPS Tensors
            #for i in range(len(self.tensors)):
            #    for j in range(len(self.tensors[i])):
            #        save_dict['tensor_{}_{}'.format(i,j)] = self.tensors[i][j].ten
            ## Add Lambda Tensors (if Canonical)
            #if self.ltensors is not None:
            #    for ind in range(len(self.ltensors)):
            #        for x in range(len(self.ltensors[ind])):
            #            for y in range(len(self.ltensors[ind][x])):
            #                save_dict['ltensor_{}_{}_{}'.format(ind,x,y)] = self.ltensors[ind][x][y].ten
            #
            #np.savez(self.fdir+self.fname,**save_dict)
            # Create file
            for i in range(5):
                try:

                    f = open_file(fdir+fname,'w')

                    # Add PEPS Info
                    create_dataset(f,'Nx',self.Nx)
                    create_dataset(f,'Ny',self.Ny)
                    create_dataset(f,'shape',self.shape)
                    create_dataset(f,'d',self.d)
                    create_dataset(f,'D',self.D)
                    create_dataset(f,'chi',self.chi)
                    create_dataset(f,'Zn',False if self.Zn is None else self.Zn)
                    create_dataset(f,'thermal',self.thermal)
                    create_dataset(f,'dZn',False if self.dZn is None else self.dZn)
                    create_dataset(f,'canonical',self.canonical)
                    create_dataset(f,'singleLayer',self.singleLayer)
                    create_dataset(f,'exact_norm_tol',self.exact_norm_tol)
                    create_dataset(f,'norm_tol',self.norm_tol)
                    create_dataset(f,'max_norm_iter',self.max_norm_iter)
                    create_dataset(f,'norm_bs_upper',self.norm_bs_upper)
                    create_dataset(f,'norm_bs_lower',self.norm_bs_lower)
                    create_dataset(f,'fname',fname)
                    create_dataset(f,'fdir',self.fdir)

                    # Add PEPS Tensors
                    for i in range(len(self.tensors)):
                        for j in range(len(self.tensors[i])):
                            
                            # Load tensor (if needed)
                            init_in_mem = self.tensors[i][j].in_mem
                            if not init_in_mem: self.tensors[i][j].from_disk()
                                
                            # Save the tensor
                            create_dataset(f,'tensor_{}_{}'.format(i,j),self.tensors[i][j].ten)
                            #create_dataset(f,'tensorlegs_{}_{}'.format(i,j),self.tensors[i][j].legs)

                            # Put the tensor back on disk (if needed)
                            if not init_in_mem: self.tensors[i][j].to_disk()

                    # Add Lambda Tensors (if Canonical)
                    if self.ltensors is not None:
                        for ind in range(len(self.ltensors)):
                            for x in range(len(self.ltensors[ind])):
                                for y in range(len(self.ltensors[ind][x])):

                                    # Load tensor (if needed)
                                    init_in_mem = self.ltensors[ind][x][y].in_mem
                                    if not init_in_mem: self.ltensors[ind][x][y].from_disk()

                                    # Save the tensor
                                    create_dataset(f,'ltensor_{}_{}_{}'.format(ind,x,y),self.ltensors[ind][x][y].ten)
                                    #create_dataset(f,'ltensorlegs_{}_{}_{}'.format(ind,x,y),self.ltensors[ind][x][y].legs)

                                    # Put the tensor back on disk (if needed)
                                    if not init_in_mem: self.ltensors[ind][x][y].to_disk()
                    # Close file
                    close_file(f)
                    break
                except:
                    #print('Saving PEPS Failed... Attempt ({}/5)'.format(i))
                    pass
        else:
            pass
            #print('Didnt save peps...')
            #raise NotImplementedError()

    def load_tensors(self,fname):
        if self.Zn is None:
            # Open File
            f = open_file(fname,'r')
            # Check to make sure this peps and the one we are loading agree
            assert(self.Nx == get_dataset(f,'Nx'))
            assert(self.Ny == get_dataset(f,'Ny'))
            # Get PEPS Tensors
            for i in range(len(self.tensors)):
                for j in range(len(self.tensors[i])):
                    self.tensors[i][j].ten = get_dataset(f,'tensor_{}_{}'.format(i,j))
            # Get Lambda Tensors (if Canonical
            if self.ltensors is not None:
                for ind in range(len(self.ltensors)):
                    for x in range(len(self.ltensors[ind])):
                        for y in range(len(self.ltensors[ind][x])):
                            self.ltensors[ind][x][y].ten = get_dataset(f,'ltensor_{}_{}_{}'.format(ind,x,y))
            # Close File
            close_file(f)
        else:
            raise NotImplementedError()

    def max_entry(self):
        maxval = 0.
        for i in range(len(self)):
            for j in range(len(self[i])):
                maxval = max(maxval,self[i][j].max_abs())
        return maxval

    def to_disk(self):
        """
        Write all peps tensors to disk
        """
        for x in range(self.Nx):
            self.col_to_disk(x)

    def from_disk(self):
        """
        Read all peps tensors from disk
        """
        for x in range(self.Nx):
            self.col_from_disk(x)

    def col_to_disk(self,x):
        """
        Write all the peps tensors in a column to disk
        """
        for y in range(self.Ny):
            self.site_to_disk(x,y)

    def col_from_disk(self,x):
        """
        Read all peps tensors in a column from disk
        """
        for y in range(self.Ny):
            self.site_from_disk(x,y)

    def row_to_disk(self,y):
        """
        Write all the peps tensors in a row to disk
        """
        for x in range(self.Nx):
            self.site_to_disk(x,y)

    def row_from_disk(self,y):
        """
        Read all peps tensors in a row from disk
        """
        for x in range(self.Nx):
            self.site_from_disk(x,y)

    def site_to_disk(self,x,y):
        """
        Write a peps tensor at site peps[x][y] to disk
        """
        self[x][y].to_disk()

    def site_from_disk(self,x,y):
        """
        Read a peps tensor at site peps[x][y] to disk
        """
        self[x][y].from_disk()
