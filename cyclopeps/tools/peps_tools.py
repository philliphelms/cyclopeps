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
from cyclopeps.tools.mps_tools import MPS,contract_mps
from cyclopeps.tools.env_tools import *
from numpy import float_
from numpy import isnan, power
import copy
from . import mps_tools

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

def flip_peps(peps):
    """
    Flip a peps horizontally

    Args:
        peps : a list of a list containing peps tensors
            The initial peps tensor

    Returns:
        peps : a list of a list containing peps tensors
            The horizontally flipped version of the peps
            tensor. This is flipped such that ...
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
            fpeps[x][y] = copy.deepcopy(peps[(Nx-1)-x][y])
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
    # Determine the correct left bond dimension
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
    ten = rand(dims,dtype=dtype)

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
    mpiprint(2, '\n[binarySearch] shape=({},{}), chi={}'.format(Nx,Ny,chi))

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
            mpiprint(0, 'binarySearch normalization exceeds max_iter... terminating')
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

def update_top_env(peps,left1,left2,right1,right2,prev_env):
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
        prev_env = ones((1,1,1,1),dtype=peps.dtype)
    tmp = einsum('LDPRU,OUuo->OLDPRuo',peps,prev_env)
    tmp = einsum('OLDPRuo,NLO->NDPRuo',tmp,left2)
    tmp = einsum('NDPRuo,nRo->NDPun',tmp,right2)
    tmp = einsum('NDPun,ldPru->NDldrn',tmp,conj(peps))
    tmp = einsum('NDldrn,MlN->MDdrn',tmp,left1)
    top_env = einsum('MDdrn,mrn->MDdm',tmp,right1)
    return top_env

def calc_top_envs(peps_col,left_bmpo,right_bmpo):
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
    Ny = len(peps_col)

    # Compute top environment
    top_env = [None]*Ny
    for row in reversed(range(Ny)):
        if row == Ny-1: prev_env = None
        else: prev_env = top_env[row+1]
        top_env[row] = update_top_env(peps_col[row],
                                      left_bmpo[2*row],
                                      left_bmpo[2*row+1],
                                      right_bmpo[2*row],
                                      right_bmpo[2*row+1],
                                      prev_env)
    return top_env

def update_bot_env(peps,left1,left2,right1,right2,prev_env):
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
        prev_env = ones((1,1,1,1),dtype=peps.dtype)
    tmp = einsum('LDPRU,MdDm->MdLPURm',peps,prev_env)
    tmp = einsum('MdLPURm,MLN->NdPURm',tmp,left1)
    tmp = einsum('NdPURm,mRn->NdPUn',tmp,right1)
    tmp = einsum('NdPUn,ldPru->NlurUn',tmp,conj(peps))
    tmp = einsum('NlurUn,NlO->OurUn',tmp,left2)
    bot_env = einsum('OurUn,nro->OuUo',tmp,right2)
    return bot_env

def calc_bot_envs(peps_col,left_bmpo,right_bmpo):
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
    Ny = len(peps_col)

    # Compute the bottom environment
    bot_env = [None]*Ny
    for row in range(Ny):
        if row == 0: prev_env = None
        else: prev_env = bot_env[row-1]
        bot_env[row] = update_bot_env(peps_col[row],
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
    (ub,sb,vb) = mps_tools.svd_ten(peps1,3,return_ent=False,return_wgt=False)
    phys_b = einsum('a,aPU->aPU',sb,vb)

    # Reduce top tensor
    peps2 = einsum('LDPRU->DPLRU',peps2)
    (ut,st,vt) = mps_tools.svd_ten(peps2,2,return_ent=False,return_wgt=False)
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
        N = einsum('UuDd->UDud',N)
        (n1,n2,n3,n4) = N.shape
        Nmat = reshape(N,(n1*n2,n3*n4))
        u,v = eigh(Nmat)
        u = pos_sqrt_vec(u)
        Nmat = einsum('ij,j,kj->ik',v,u,v)
        N = reshape(Nmat,(n1,n2,n3,n4))
        N = einsum('UDud->UuDd',N)

        # Check to ensure N is positive
        if DEBUG:
            Ntmp = copy.deepcopy(N)
            Ntmp = einsum('UuDd->UDud',Ntmp)
            (n1_,n2_,n3_,n4_) = Ntmp.shape
            Ntmp = reshape(Ntmp,(n1_*n2_,n3_*n4_))
            mpiprint(0,'This makes N positive, but not every element of N positive??:\n{}'.format(Ntmp))

    return N

def calc_local_env(peps1,peps2,env_top,env_bot,lbmpo,rbmpo,reduced=True,hermitian=True,positive=True):
    """
    Calculate the local environment around two peps tensors

    Args:
        peps1 : peps tensor
            The peps tensor for the bottom site
        peps2 : peps tensor
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
        if DEBUG:
            # Calculate initial norm exactly
            tmp = einsum('abcd,aef->febcd',env_bot,lbmpo[0])
            tmp = einsum('febcd,ecghi->fbgihd',tmp,peps1)
            tmp = einsum('fbgihd,dhj->fbgij',tmp,rbmpo[0])
            tmp = einsum('fbgij,fkl->lkbgij',tmp,lbmpo[1])
            tmp = einsum('lkbgij,kbgmn->lnmij',tmp,peps1)
            tmp = einsum('lnmij,jmo->lnio',tmp,rbmpo[1])
            tmp = einsum('lnio,lpq->qpnio',tmp,lbmpo[2])
            tmp = einsum('qpnio,pirst->qnrtso',tmp,peps2)
            tmp = einsum('qnrtso,osu->qnrtu',tmp,rbmpo[2])
            tmp = einsum('qnrtu,qvw->wvnrtu',tmp,lbmpo[3])
            tmp = einsum('wvnrtu,vnrxy->wyxtu',tmp,peps2)
            tmp = einsum('wyxtu,uxz->wytz',tmp,rbmpo[3])
            norm =einsum('wytz,wytz->',tmp,env_top)
            mpiprint(0,'Tedious norm = {}'.format(norm))

        # Get reduced tensors
        ub,phys_b,phys_t,vt = reduce_tensors(peps1,peps2)

        # Compute bottom half of environment
        tmp = einsum('CDdc,ClB->BlDdc',env_bot,lbmpo[0])
        tmp = einsum('BlDdc,ldru->BDurc',tmp,conj(ub))
        tmp = einsum('BDurc,crb->BDub',tmp,rbmpo[0])
        tmp = einsum('BDub,BLA->ALDub',tmp,lbmpo[1])
        tmp = einsum('ALDub,LDRU->AURub',tmp,ub)
        envt= einsum('AURub,bRa->AUua',tmp,rbmpo[1])

        # Compute top half of environment
        tmp = einsum('CUuc,BLC->BLUuc',env_top,lbmpo[3])
        tmp = einsum('BLUuc,LDRU->BDRuc',tmp,vt)
        tmp = einsum('BDRuc,bRc->BDub',tmp,rbmpo[3])
        tmp = einsum('BDub,AlB->AlDub',tmp,lbmpo[2])
        tmp = einsum('AlDub,ldru->ADdrb',tmp,conj(vt))
        envb= einsum('ADdrb,arb->ADda',tmp,rbmpo[2])

        # Compute Environment
        N = einsum('AUua,ADda->UuDd',envt,envb)
        N = make_N_positive(N,hermitian=hermitian,positive=positive)

        return ub,phys_b,phys_t,vt,N
    else:
        mpiprint(0,'Only reduced update implemented')
        import sys
        sys.exit()

def calc_local_op(phys_b_bra,phys_t_bra,N,ham,
                      phys_b_ket=None,phys_t_ket=None,
                      reduced=True,normalize=True):
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
        tmp1= einsum('APU,UQB->APQB',phys_b_bra,phys_t_bra)
        tmp1 = einsum('APQB,AaBb->aPQb',tmp1,N)
        tmp2= einsum('apu,uqb->apqb',phys_b_ket,phys_t_ket)
        tmp = einsum('aPQb,apqb->PQpq',tmp1,tmp2)
        if ham is not None:
            E = einsum('PQpq,PQpq->',tmp,ham)
        else:
            E = einsum('PQPQ->',tmp)
        norm = einsum('PQPQ->',tmp)
        mpiprint(7,'E = {}/{} = {}'.format(E,norm,E/norm))
        if normalize:
            return E/norm
        else:
            return E
    else:
        mpiprint(0,'Only reduced update implemented')
        import sys
        sys.exit()

def calc_N(row,peps_col,left_bmpo,right_bmpo,top_envs,bot_envs,hermitian=True,positive=True):

    if row == 0:
        if len(peps_col) == 2:
            # Only two sites in column, use identity at both ends
            ub,phys_b,phys_t,vt,N = calc_local_env(peps_col[row],
                                             peps_col[row+1],
                                             ones((1,1,1,1),dtype=top_envs[0].dtype),
                                             ones((1,1,1,1),dtype=top_envs[0].dtype),
                                             left_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                             right_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                             hermitian=hermitian,
                                             positive=positive)
        else:
            # Get the local environment tensor
            ub,phys_b,phys_t,vt,N = calc_local_env(peps_col[row],
                                             peps_col[row+1],
                                             top_envs[row+2],
                                             ones((1,1,1,1),dtype=top_envs[0].dtype),
                                             left_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                             right_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                             hermitian=hermitian,
                                             positive=positive)
    elif row == len(peps_col)-2:
        ub,phys_b,phys_t,vt,N = calc_local_env(peps_col[row],
                                         peps_col[row+1],
                                         ones((1,1,1,1),dtype=top_envs[0].dtype),
                                         bot_envs[row-1],
                                         left_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                         right_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                         hermitian=hermitian,
                                         positive=positive)
    else:
        # Get the local environment tensor
        ub,phys_b,phys_t,vt,N = calc_local_env(peps_col[row],
                                         peps_col[row+1],
                                         top_envs[row+2],
                                         bot_envs[row-1],
                                         left_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                         right_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                                         hermitian=hermitian,
                                         positive=positive)
    return ub,phys_b,phys_t,vt,N

def calc_single_column_op(peps_col,left_bmpo,right_bmpo,ops_col,normalize=True):
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
    top_envs = calc_top_envs(peps_col,left_bmpo,right_bmpo)
    bot_envs = calc_bot_envs(peps_col,left_bmpo,right_bmpo)

    # Calculate Energy
    E = zeros(len(ops_col))
    for row in range(len(ops_col)):
        _,phys_b,phys_t,_,N = calc_N(row,peps_col,left_bmpo,right_bmpo,top_envs,bot_envs,hermitian=False,positive=False)
        E[row] = calc_local_op(phys_b,phys_t,N,ops_col[row],normalize=normalize)
    return E

def calc_all_column_op(peps,ops,chi=10,return_sum=True,normalize=True):
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

    Returns:
        val : float
            The contribution of the column's interactions to
            the observable's expectation value
    """

    # Figure out peps size
    Nx = len(peps)
    Ny = len(peps[0])

    # Compute the boundary MPOs
    right_bmpo = calc_right_bound_mpo(peps, 0,chi=chi,return_all=True)
    left_bmpo  = calc_left_bound_mpo (peps,Nx,chi=chi,return_all=True)
    ident_bmpo = mps_tools.identity_mps(len(right_bmpo[0]),dtype=peps[0][0].dtype)

    # Loop through all columns
    E = zeros((len(ops),len(ops[0])),dtype=peps[0][0].dtype)
    for col in range(Nx):
        if col == 0:
            E[col,:] = calc_single_column_op(peps[col],ident_bmpo,right_bmpo[col],ops[col],normalize=normalize)
        elif col == Nx-1:
            # Use Identity on the right side
            E[col,:] = calc_single_column_op(peps[col],left_bmpo[col-1],ident_bmpo,ops[col],normalize=normalize)
        else:
            E[col,:] = calc_single_column_op(peps[col],left_bmpo[col-1],right_bmpo[col],ops[col],normalize=normalize)
    mpiprint(8,'Energy [:,:] = \n{}'.format(E))

    if return_sum:
        return summ(E)
    else:
        return E

def calc_peps_op(peps,ops,chi=10,return_sum=True,normalize=True):
    """
    Calculate the expectation value for a given operator

    Args:
        peps : A list of lists of peps tensors
            The PEPS to be normalized
        ops :
            The operator to be contracted with the peps

    Kwargs:
        chi : int
            The maximum bond dimension for the boundary mpo

    Returns:
        val : float
            The resulting observable's expectation value
    """
    # Absorb Lambda tensors if needed
    try:
        peps = peps_absorb_lambdas(peps.tensors,peps.ltensors,mk_copy=True)
    except:
        pass

    # Calculate contribution from interactions between columns
    col_energy = calc_all_column_op(peps,ops[0],chi=chi,normalize=normalize)

    # Calculate contribution from interactions between rows
    peps = rotate_peps(peps,clockwise=True)
    row_energy = calc_all_column_op(peps,ops[1],chi=chi,normalize=normalize)
    peps = rotate_peps(peps,clockwise=False)
    
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

def increase_peps_mbd(peps,Dnew,noise=0.01):
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

    # Get unitary tensor for insertion
    identity = zeros((Dnew,Dold),dtype=peps[0][0].dtype)
    identity[:Dold,:] = eye(Dold,dtype=peps[0][0].dtype)
    mat = identity + noise*rand((Dnew,Dold),dtype=peps[0][0].dtype)
    mat = svd(mat)[0]

    # Loop through all peps tensors
    for col in range(Nx):
        for row in range(Ny):
            # Increase left bond
            if row != 0:
                peps[row][col] = einsum('Ll,ldpru->Ldpru',mat,peps[row][col])
            # Increase down bond
            if col != 0:
                peps[row][col] = einsum('Dd,ldpru->lDpru',mat,peps[row][col])
            # Increase right bond
            if row != Nx-1:
                peps[row][col] = einsum('Rr,ldpru->ldpRu',mat,peps[row][col])
            # Increase up bond
            if col != Ny-1:
                peps[row][col] = einsum('Uu,ldpru->ldprU',mat,peps[row][col])
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
                 singleLayer=True,max_norm_iter=100,
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
        return calc_left_bound_mpo(peps,col,chi=chi,singleLayer=singleLayer,truncate=truncate,return_all=return_all)

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
        return calc_right_bound_mpo(peps,col,chi=chi,singleLayer=singleLayer,truncate=truncate,return_all=return_all)

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

    def calc_op(self,ops,chi=None,normalize=True):
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

        Returns:
            val : float
                The resulting observable's expectation value
        """
        if chi is None: chi = self.chi
        # Calculate the operator's value
        return calc_peps_op(self,ops,chi=chi,normalize=normalize)

    def increase_mbd(self,newD,chi=None,noise=0.01):
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

    def flip(self):
        """
        Flip the peps columns
        """
        self.tensors = flip_peps(self.tensors)
        self.ltensors= flip_lambda(self.ltensors)
