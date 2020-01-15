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

from cyclopeps.tools.params import *
from cyclopeps.tools.utils import *
from cyclopeps.tools.mps_tools import *
from numpy import float_
from numpy import isnan, power
import copy

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PEPS ENVIRONMENT FUNCTIONS 
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
    _,_,d,D,_ = bra[0].shape

    # Copy the ket column if needed
    if ket is None:
        ket = copy.deepcopy(bra)

    # Make list to hold resulting mpo
    bound_mpo = []

    for row in range(Ny):
        # Add Bra-ket contraction
        res = einsum('ldpru,LDpRU->lLdDRurU',ket[row],bra[row])
        # Reshape so it is an MPO
        (Dl,Dd,Dp,Dr,Du) = bra[row].shape
        res = reshape(res,(Dl*Dl*Dd*Dd,Dr,Dr*Du*Du))
        # Append to boundary_mpo
        bound_mpo.append(res)

        # Add correct identity
        I1 = eye(Dr)
        I2 = eye(Du)
        I3 = eye(Du)
        I = einsum('du,DU,lr->dlDruU',I1,I2,I3)
        # Reshape so it is an MPO
        I = reshape(I,(Du*Dr*Du,Dr,Du*Du))
        bound_mpo.append(I)

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

    for row in range(Ny):
        mpiprint(5,'Adding Site {} to Ket'.format(row))

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
        if DEBUG:
            mpiprint(6,'Computing initial norm')
            norm0 = bound_mps.norm()
        mpiprint(6,'Applying SVD')
        bound_mps.apply_svd(chi)
        if DEBUG:
            mpiprint(6,'Computing Resulting norm')
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

    for row in range(Ny):
        mpiprint(5,'Adding Site {} to bra'.format(row))

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
        if DEBUG:
            mpiprint(6,'Computing initial norm')
            norm0 = bound_mps.norm()
        mpiprint(6,'Applying SVD')
        bound_mps.apply_svd(chi)
        if DEBUG:
            mpiprint(6,'Computing Resulting norm')
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
    mpiprint(3,'Updating boundary mpo (sl), maxelem = {}'.format(bound_mpo.max_elem()))
    # Find size of peps column and dims of tensors
    Ny = len(bra)
    _,_,d,D,_ = bra[0].shape
    
    # Copy the ket column if needed
    if ket is None:
        ket = copy.deepcopy(bra)

    # First Layer (ket) #####################################
    bound_mpo = left_bmpo_sl_add_ket(ket,bound_mpo,D,Ny,chi=chi,truncate=truncate)
    # Second Layer (bra) ####################################
    bound_mpo = left_bmpo_sl_add_bra(bra,bound_mpo,D,Ny,chi=chi,truncate=truncate)

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
    import sys
    sys.exit()

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
                rpeps[y][Nx-1-x] = copy.deepcopy(peps[x][y])
                # Reorder Indices
                rpeps[y][Nx-1-x] = transpose(rpeps[y][Nx-1-x],[1,3,2,4,0])
            else:
                # Copy Correct Tensor
                rpeps[Ny-1-y][x] = copy.deepcopy(peps[x][y])
                # Reorder Indices
                rpeps[Ny-1-y][x] = transpose(rpeps[Ny-1-y][x],[4,0,2,1,3])

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
                    tmp += [copy.deepcopy(Lambda[1][Ny-2-y][x])]
                else:
                    tmp += [copy.deepcopy(Lambda[1][y][Nx-1-x])]
            vert += [tmp]

        # Lambda tensors along horizontal bonds
        horz = []
        for x in range(Nx-1):
            tmp = []
            for y in range(Ny):
                if clockwise:
                    tmp += [copy.deepcopy(Lambda[0][Ny-1-y][x])]
                else:
                    tmp += [copy.deepcopy(Lambda[0][y][Nx-2-x])]
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
            if mk_copy:
                fpeps[x][y] = copy.deepcopy(peps[(Nx-1)-x][y])
            else:
                fpeps[x][y] = peps[(Nx-1)-x][y]
            # Reorder Indices
            fpeps[x][y] = transpose(fpeps[x][y],[3,1,2,0,4])

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
                tmp += [copy.deepcopy(Lambda[0][(Nx-1)-x][y])]
            vert += [tmp]
        # Lambda tensors along horizontal bonds
        horz = []
        for x in range(Nx-1):
            tmp = []
            for y in range(Ny):
                tmp += [copy.deepcopy(Lambda[1][(Nx-2)-x][y])]
            horz += [tmp]

        # Add to tensors
        fLambda = [vert,horz]

        # Return Flipped peps
        return fLambda
    else:
        return None

def peps_col_to_mps(peps_col,mk_copy=True):
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

    Kwargs:
        mk_copy : bool
            Specify whether a copy of the peps col should be made
            and put into the MPS. Default is True, meaning a copy
            is made.

    Returns:
        mps : 1D Array
            The resulting 1D array containing the PEPS column's tensor

    """

    # Determine number of rows
    Ny = len(peps_col)

    # Make copy (if wanted)
    if mk_copy: peps_col = copy.deepcopy(peps_col)
    # Create a list to hold the copy
    peps_col_copy = [None]*Ny
    for row in range(Ny):
        # Copy the tensor
        (Dl,Dd,d,Dr,Du) = peps_col[row].shape
        # Transpose to put left, physical, and right bonds in middle
        peps_col[row] = transpose(peps_col[row],axes=[1,0,2,3,4])
        # Reshape (to lump left, physical, and right tensors)
        peps_col[row] = reshape(peps_col[row],(Dd,Dl*d*Dr,Du))

    # Convert PEPS column into an MPS
    mps = MPS()
    mps.input_mps_list(peps_col)

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

def rand_peps_tensor(Nx,Ny,x,y,d,D,dtype=float_):
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

    # Create a random tensor
    dims = (Dl,Dd,d,Dr,Du)
    #ten = rand(dims,dtype=dtype)
    ten = 0.95*ones(dims,dtype=dtype)+0.1*rand(dims,dtype=dtype)

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
    """

    Nx = len(peps)
    Ny = len(peps[0])
    for xind in range(Nx):
        for yind in range(Ny):
            peps[xind][yind] *= const
    return peps

def normalize_peps(peps,max_iter=100,norm_tol=20,chi=4,up=1.0,
                    down=0.0,singleLayer=True):
    """
    Normalize the full PEPS by doing a binary search on the
    interval [down, up] for the factor which, when multiplying
    every element of the PEPS tensors, yields a rescaled PEPS
    with norm equal to 1.0.

    Args:
        peps : List
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
            The normalized version of the PEPS, stored as a
            list of lists

    """

    # Figure out peps size
    Nx = peps.Nx
    Ny = peps.Ny

    pwr = -1.0 / (2*Nx*Ny) # NOTE: if trying to use this procedure to 
                           # normalize a partition function, remove
                           # the factor of 2 in this denominator
    mpiprint(4, '\n[binarySearch] shape=({},{}), chi={}'.format(Nx,Ny,chi))

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
            if z < 10.0**(-1*norm_tol) or z > 10.0**(norm_tol) or isnan(z):
                if z > 1.0 or isnan(z):
                    up = scale
                    scale = (up+down)/2.0
                else:
                    down = scale
                    scale = (up+down)/2.0
            # close to convergence, apply "exact" scale
            else:
                sfac = power(z,pwr)
                scale = sfac*scale
                mpiprint(2, 'apply exact scale: {}'.format(scale))

        if istep == max_iter:
            mpiprint(4, 'binarySearch normalization exceeds max_iter... terminating')
            print('Exceeded normalization!')
            break

        peps_try = multiply_peps_elements(peps.copy(),scale)

    return z, peps_try

def calc_peps_norm(peps,chi=4,singleLayer=True):
    """
    Calculate the norm of the PEPS

    Args:
        peps : List
            A list of a list of tensors, corresponding to
            the PEPS for which we will compute the norm

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
    # TODO Add - separate bra and ket
    # Absorb Lambda tensors if needed
    try:
        peps = peps_absorb_lambdas(peps.tensors,peps.ltensors,mk_copy=True)
    except:
        pass

    # Get PEPS Dims
    Nx = len(peps)
    Ny = len(peps[0])

    # Get the boundary MPO from the left (for the furthest right column)
    left_bound_mpo  = calc_left_bound_mpo(peps,Nx,chi=chi,singleLayer=singleLayer)

    # Get the boundary MPO from the right (for the furthest right column)
    right_bound_mpo = calc_right_bound_mpo(peps,Nx-2,chi=chi,singleLayer=singleLayer)

    # Contract the two MPOs
    norm = contract_mps(left_bound_mpo,right_bound_mpo)

    # Return result
    return norm

def make_rand_peps(Nx,Ny,d,D,canonical=False,dtype=float_):
    """
    Make a random PEPS
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
            tensors[x][y] = rand_peps_tensor(Nx,Ny,x,y,d,D,dtype=dtype)
        # At the end of each column, make the norm smaller
        tensors[x][:] = normalize_peps_col(tensors[x][:])

    return tensors

def make_rand_lambdas(Nx,Ny,D,dtype=float_):
    """
    Make random diagonal matrices to serve as the
    singular values for the Gamma-Lambda canonical
    form of the PEPS

    Note:
    Used primarily for the simple update contraction scheme
    """

    # Lambda tensors along vertical bonds
    vert = []
    for x in range(Nx):
        tmp = []
        for y in range(Ny-1):
            tmp += [rand((D),dtype=dtype)]
        vert += [tmp]

    # Lambda tensors along horizontal bonds
    horz = []
    for x in range(Nx-1):
        tmp = []
        for x in range(Ny):
            tmp += [rand((D),dtype=dtype)]
        horz += [tmp]

    # Add horizontal and vertical lambdas to tensor list
    tensors = [vert,horz]
    return tensors

def update_top_env(bra,ket,left1,left2,right1,right2,prev_env):
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


     Where the tensor with legs :
        "aehmo" is the left boundary tensor
        "eijfb" is the peps tensor
        "mpjnk" is the conj of the peps tensor
        "dglnq" is the right boundary tensor
        "abcd" is the previous top enviromnent
        "oipq" is the new top environment

    """
    if prev_env is None:
        prev_env = ones((1,1,1,1),dtype=bra.dtype)
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


     Where the tensor with legs :
        "aehmo" is the left boundary tensor
        "eijfb" is the peps tensor
        "mpjnk" is the conj of the peps tensor
        "dglnq" is the right boundary tensor
        "abcd" is the previous top enviromnent
        "oipq" is the new top environment

    """

    # Figure out height of peps column
    Ny = len(bra_col)

    # Copy bra if needed
    if ket_col is None: 
        ket_col = copy.deepcopy(bra_col)
    # TODO - Conjugate this ket col

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

     +-------+-------+-------+
     |       |       |       |
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
        prev_env = ones((1,1,1,1),dtype=bra.dtype)
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

     +-------+-------+-------+
     |       |       |       |
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
        ket_col = copy.deepcopy(bra_col)
    # TODO - Conjugate this ket column

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
    peps1 = einsum('LDPRU->LDRPU',peps1)
    (ub,sb,vb) = svd_ten(peps1,3,return_ent=False,return_wgt=False)
    phys_b = einsum('a,aPU->aPU',sb,vb)

    # Reduce top tensor
    peps2 = einsum('LDPRU->DPLRU',peps2)
    (ut,st,vt) = svd_ten(peps2,2,return_ent=False,return_wgt=False)
    phys_t = einsum('DPa,a->DPa',ut,st)
    vt = einsum('aLRU->LaRU',vt)

    if DEBUG:
        # Check to make sure initial and reduced peps tensors are identical
        final = einsum('LDRa,aPb,bpc,lcru->lLDPRpru',ub,phys_b,phys_t,vt)
        mpiprint(0,'Reduced Difference = {}'.format(summ(abss(original-final))))

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
        N1 = copy.deepcopy(N)
        N1 = einsum('UuDd->UDud',N1) # Could be UduD and uDUd instead
        N = einsum('UuDd->udUD',N)
        N = (N+N1)/2.
        N1 = copy.deepcopy(N)
        N = einsum('UDab,abud->UuDd',N,N1)

        # Check to ensure N is hermitian
        if DEBUG:
            Ntmp = copy.deepcopy(N)
            Ntmp = einsum('UuDd->UDud',Ntmp)
            (n1_,n2_,n3_,n4_) = Ntmp.shape
            Ntmp = reshape(Ntmp,(n1_*n2_,n3_*n4_))
            mpiprint(0,'Check if this N is hermitian:\n{}'.format(Ntmp))

    # Get a positive approximation of the environment
    if positive:
        try:
            N = einsum('UuDd->UDud',N)
            (n1,n2,n3,n4) = N.shape
            Nmat = reshape(N,(n1*n2,n3*n4))
            u,v = eigh(Nmat)
            u = pos_sqrt_vec(u)
            Nmat = einsum('ij,j,kj->ik',v,u,v)
            N = reshape(Nmat,(n1,n2,n3,n4))
            N = einsum('UDud->UuDd',N)
        except Exception as e:
            print('Failed to make N positive, eigenvalues did not converge')

        # Check to ensure N is positive
        if DEBUG:
            Ntmp = copy.deepcopy(N)
            Ntmp = einsum('UuDd->UDud',Ntmp)
            (n1_,n2_,n3_,n4_) = Ntmp.shape
            Ntmp = reshape(Ntmp,(n1_*n2_,n3_*n4_))
            mpiprint(0,'This makes N positive, but not every element of N positive??:\n{}'.format(Ntmp))

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
        tmp = einsum('CdDc,CLB->BLdDc',env_bot,lbmpo[0])
        tmp = einsum('BLdDc,LDRU->BdURc',tmp,ub)
        tmp = einsum('BdURc,cRb->BdUb',tmp,rbmpo[0])
        tmp = einsum('BdUb,BlA->AldUb',tmp,lbmpo[1])
        tmp = einsum('AldUb,ldru->AurUb',tmp,ubk)
        envb= einsum('AurUb,bra->AuUa',tmp,rbmpo[1])

        # Compute top half of environment
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
        mpiprint(0,'Only reduced update implemented')
        import sys
        sys.exit()

def calc_local_op(phys_b_bra,phys_t_bra,N,ham,
                      phys_b_ket=None,phys_t_ket=None,
                      reduced=True,normalize=True,return_norm=False):
    """
    Calculate the normalized Energy of the system
    """
    # Make some copies
    if phys_t_ket is None:
        phys_t_ket = conj(copy.deepcopy(phys_t_bra))
    if phys_b_ket is None:
        phys_b_ket = conj(copy.deepcopy(phys_b_bra))

    # Compute Energy (or op value
    if reduced:
        tmp = einsum('APU,UQB->APQB',phys_b_bra,phys_t_bra)
        tmp1= einsum('APQB,aAbB->aPQb',tmp,N)
        tmp2= einsum('apu,uqb->apqb',phys_b_ket,phys_t_ket)
        tmp = einsum('aPQb,apqb->PQpq',tmp1,tmp2)
        if ham is not None:
            E = einsum('PQpq,PQpq->',tmp,ham)
        else:
            E = einsum('PQPQ->',tmp)
        norm = einsum('PQPQ->',tmp)
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
        mpiprint(0,'Only reduced update implemented')
        import sys
        sys.exit()

def calc_N(row,bra_col,left_bmpo,right_bmpo,top_envs,bot_envs,hermitian=True,positive=True,ket_col=None):
    """
    Calculate the environment tensor
    """
    # Copy bra if needed
    _ket_col = ket_col
    if ket_col is None: 
        ket_col = copy.deepcopy(bra_col)
    # TODO - Conjugate this ket column

    if row == 0:
        if len(bra_col) == 2:
            # Only two sites in column, use identity at both ends
            res = calc_local_env(bra_col[row],
                                 bra_col[row+1],
                                 ket_col[row],
                                 ket_col[row+1],
                                 ones((1,1,1,1),dtype=top_envs[0].dtype),
                                 ones((1,1,1,1),dtype=top_envs[0].dtype),
                                 left_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                 right_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                 hermitian=hermitian,
                                 positive=positive)
        else:
            # Get the local environment tensor
            res = calc_local_env(bra_col[row],
                                 bra_col[row+1],
                                 ket_col[row],
                                 ket_col[row+1],
                                 top_envs[row+2],
                                 ones((1,1,1,1),dtype=top_envs[0].dtype),
                                 left_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                 right_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                 hermitian=hermitian,
                                 positive=positive)
    elif row == len(bra_col)-2:
        res = calc_local_env(bra_col[row],
                             bra_col[row+1],
                             ket_col[row],
                             ket_col[row+1],
                             ones((1,1,1,1),dtype=top_envs[0].dtype),
                             bot_envs[row-1],
                             left_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                             right_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                             hermitian=hermitian,
                             positive=positive)
    else:
        # Get the local environment tensor
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
    E = zeros(len(ops_col))
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
        ket : PEPS Object
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
    ident_bmpo = identity_mps(len(right_bmpo[0]),dtype=peps[0][0].dtype)

    # Loop through all columns
    E = zeros((len(ops),len(ops[0])),dtype=peps[0][0].dtype)
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
        return summ(E)
    else:
        return E

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
    try:
        peps = peps_absorb_lambdas(peps.tensors,peps.ltensors,mk_copy=True)
    except: pass
    try:
        ket = peps_absorb_lambdas(ket.tensors,ket.ltensors,mk_copy=True)
    except: pass

    # Calculate contribution from interactions between columns
    col_energy = calc_all_column_op(peps,ops[0],chi=chi,normalize=normalize,return_sum=return_sum,ket=ket)

    # Calculate contribution from interactions between rows
    peps = rotate_peps(peps,clockwise=True)
    if ket is not None: 
        ket = rotate_peps(ket,clockwise=True)
    row_energy = calc_all_column_op(peps,ops[1],chi=chi,normalize=normalize,return_sum=return_sum,ket=ket)
    peps = rotate_peps(peps,clockwise=False)
    if ket is not None: 
        ket = rotate_peps(ket,clockwise=False)

    # Return Result
    if return_sum:
        return summ(col_energy)+summ(row_energy)
    else:
        return col_energy,row_energy

def increase_peps_mbd_lambda(Lambda,Dnew,noise=0.01):
    """
    Increase the bond dimension of lambda tensors in a 
    canonical peps

    Args:
        Lambda : 3D array
            Lists of lambda tensors for the canonical peps
        Dnew : int
            The new bond dimension

    Kwargs:
        noise : float
            The maximum magnitude of random noise to be incorporated
            in increasing the bond dimension
    
    Returns:
        Lambda : 3D array
            Lists of lambda tensors with increased bond dimensions
    """
    if Lambda is not None:
        # Figure out peps size
        Nx = len(Lambda[0])
        Ny = len(Lambda[0][0])
        Dold = Lambda[0][0][0].shape[0]

        # Get unitary tensor for insertion
        identity = zeros((Dnew,Dold),dtype=Lambda[0][0][0].dtype)
        identity[:Dold,:] = eye(Dold,dtype=Lambda[0][0][0].dtype)
        mat = identity + noise*rand((Dnew,Dold),dtype=Lambda[0][0][0].dtype)
        mat = svd(mat)[0]

        # Loop through all possible tensors and increase their sizes
        for ind in range(len(Lambda)):
            for x in range(len(Lambda[ind])):
                for y in range(len(Lambda[ind][x])):
                    Lambda[ind][x][y] = einsum('Ll,l->L',mat,Lambda[ind][x][y])

        # Return result
        return Lambda
    else: 
        return None

def increase_peps_mbd(peps,Dnew,noise=1e-10):
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
    # Figure out peps size
    Nx = len(peps)
    Ny = len(peps[0])
    Dold = peps[0][0].shape[3]
    
    for col in range(Nx):
        for row in range(Ny):
            # Determine tensor shape
            old_shape = list(peps[row][col].shape)
            new_shape = list(peps[row][col].shape)
            # Increase left bond dimension
            if row != 0:
                new_shape[0] = Dnew
            if col != 0:
                new_shape[1] = Dnew
            if row != Nx-1:
                new_shape[3] = Dnew
            if col != Ny-1:
                new_shape[4] = Dnew
            # Create an empty tensor
            ten = zeros(new_shape,dtype=peps[row][col].dtype)
            ten[:old_shape[0],:old_shape[1],:old_shape[2],:old_shape[3],:old_shape[4]] = peps[row][col].copy()
            # Add some noise (if needed
            ten_noise = noise*rand(new_shape,dtype=peps[row][col].dtype)
            ten += ten_noise
            # Put new tensor back into peps
            peps[row][col] = ten

    # Return result
    return peps

def copy_peps_tensors(peps):
    """
    Create a copy of the PEPS tensors
    """
    copy = []
    for x in range(len(peps)):
        tmp = []
        for y in range(len(peps[0])):
            tmp += [copy.deepcopy(peps[x][y])]
        copy += [tmp]
    return copy

def peps_absorb_lambdas(Gamma,Lambda,mk_copy=False):
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
                # Absorb left lambda
                if x is not 0:
                    Gamma[x][y] = einsum('ldpru,l->ldpru',Gamma[x][y],sqrt(Lambda[1][x-1][y]))
                # Absorb down lambda
                if y is not 0:
                    Gamma[x][y] = einsum('ldpru,d->ldpru',Gamma[x][y],sqrt(Lambda[0][x][y-1]))
                # Absorb right lambda
                if x is not (Nx-1):
                    Gamma[x][y] = einsum('ldpru,r->ldpru',Gamma[x][y],sqrt(Lambda[1][x][y]))
                # Absorb up lambda
                if y is not (Ny-1):
                    Gamma[x][y] = einsum('ldpru,u->ldpru',Gamma[x][y],sqrt(Lambda[0][x][y]))

    # Return results
    return Gamma

# -----------------------------------------------------------------
# PEPS Class

class PEPS:
    """
    A class to hold and manipulate PEPS
    """

    def __init__(self,Nx=10,Ny=10,d=2,D=2,
                 chi=None,norm_tol=20,canonical=False,
                 singleLayer=True,max_norm_iter=50,
                 norm_BS_upper=1.0,norm_BS_lower=0.0,
                 norm_BS_print=1,dtype=float_,normalize=True):
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
            norm_tol : float
                How close to 1. the norm should be before exact
                artihmetic is used in the normalization procedure.
                See documentation of normalize_peps() function
                for more details.
            canonical : bool
                A PEPS in the Gamma Lambda formalism, with diagonal
                matrices between each set of PEPS tensors
            singleLayer : bool
                Whether to use a single layer environment
                (currently only option implemented)
            max_norm_iter : int
                The maximum number of normalization iterations
            norm_BS_upper : float
                The upper bound for the binary search factor
                during normalization.
            norm_BS_lower : float
                The lower bound for the binary search factor
                during normalization.
            norm_BS_print : boolean
                Controls output of binary search normalization
                procedure.
            dtype : dtype
                The data type for the PEPS
            normalize : bool
                Whether the initial random peps should be normalized

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
        if chi is None: chi = D**2
        self.chi         = chi
        self.norm_tol    = norm_tol
        self.canonical   = canonical
        self.singleLayer = singleLayer
        self.max_norm_iter = max_norm_iter
        self.dtype       = dtype
        self.norm_BS_upper = norm_BS_upper
        self.norm_BS_lower = norm_BS_lower
        self.norm_BS_print = norm_BS_print

        # Make a random PEPS
        self.tensors = make_rand_peps(self.Nx,
                                      self.Ny,
                                      self.d,
                                      self.D,
                                      dtype=self.dtype)

        # Add in lambda "singular value" matrices
        if self.canonical:
            self.ltensors = make_rand_lambdas(self.Nx,
                                              self.Ny,
                                              self.D,
                                              self.dtype)
        else:
            self.ltensors = None

        # Normalize the PEPS
        if normalize:
            self.normalize()

    def calc_bmpo_left(self,col,chi=4,singleLayer=True,truncate=True,return_all=False):
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
        if chi is None:
            chi = self.chi
        if singleLayer is None:
            singleLayer = self.singleLayer
        return calc_left_bound_mpo(self,col,chi=chi,singleLayer=singleLayer,truncate=truncate,return_all=return_all)

    def calc_bmpo_right(self,col,chi=None,singleLayer=None,truncate=True,return_all=False):
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
        if chi is None:
            chi = self.chi
        if singleLayer is None:
            singleLayer = self.singleLayer
        return calc_right_bound_mpo(self,col,chi=chi,singleLayer=singleLayer,truncate=truncate,return_all=return_all)

    def calc_norm(self,chi=None,singleLayer=None):
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

        Returns:
            norm : float
                The (approximate) norm of the PEPS
        """
        if chi is None: chi = self.chi
        if singleLayer is None: singleLayer = self.singleLayer
        return calc_peps_norm(self,chi=chi,singleLayer=singleLayer)

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
        if up is None: up = self.norm_BS_upper
        if down is None: down = self.norm_BS_lower
        if singleLayer is None: singleLayer = self.singleLayer
        # Run the normalization procedure
        norm, self.tensors = normalize_peps(self,
                                      max_iter = max_iter,
                                      norm_tol = norm_tol,
                                      chi = chi,
                                      up = up,
                                      down = down,
                                      singleLayer=singleLayer)

        return norm

    def calc_op(self,ops,chi=None,normalize=True,return_sum=True,ket=None):
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

        Returns:
            val : float
                The resulting observable's expectation value
        """
        if chi is None: chi = self.chi
        # Calculate the operator's value
        return calc_peps_op(self,ops,chi=chi,normalize=normalize,return_sum=return_sum,ket=ket)

    def increase_mbd(self,newD,chi=None,noise=0.01,normalize=True):
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
        self.tensors = increase_peps_mbd(self.tensors,newD,noise=noise)
        self.ltensors = increase_peps_mbd_lambda(self.ltensors,newD,noise=noise)
        self.normalize()

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
                         max_norm_iter=self.max_norm_iter,
                         norm_BS_upper=self.norm_BS_upper,
                         norm_BS_lower=self.norm_BS_lower,
                         dtype=self.dtype,normalize=False)

        # Copy peps tensors
        for i in range(self.Nx):
            for j in range(self.Ny):
                peps_copy.tensors[i][j] = copy.deepcopy(self.tensors[i][j])

        # Copy lambda tensors (if there)
        if self.ltensors is not None:
            for ind in range(len(self.ltensors)):
                for x in range(len(self.ltensors[ind])):
                    for y in range(len(self.ltensors[ind][x])):
                        peps_copy.ltensors[ind][x][y] = copy.deepcopy(self.ltensors[ind][x][y])

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
