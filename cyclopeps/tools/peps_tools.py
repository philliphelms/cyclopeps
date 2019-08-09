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
import copy

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
    for row in range(Ny):
        mps[row] = peps_col[row]

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

def normalize_peps(peps,max_iter=100,norm_tol=1e2,change_int=1e-2,chi=4,singleLayer=True):
    """
    Normalize the full PEPS

    Args:
        peps : List
            The PEPS to be normalized, stored as a list of lists

    Kwargs:
        max_iter : int
            The maximum number of iterations of the normalization 
            procedure. Default is 100.
        norm_tol : float
            How near 1. the norm must be for the normalization 
            procedure to be considered to be converged. Default
            is 100.
        change_int : float
            The magnitude with which we change entries in the 
            PEPS tensor to try to get it closer to 1. The default
            is 1e-2.
        chi : int
            Boundary MPO maximum bond dimension
        single_layer : bool
            Indicates whether to use a single layer environment
            (currently it is the only option...)

    Returns:
        norm : float
            The approximate norm of the PEPS after the normalization
            procedure
        peps : list
            The renormalized version of the PEPS, stored as a 
            list of lists
    """
    # Figure out peps size
    Nx = len(peps)
    Ny = len(peps[0])

    # Calculate the norm
    norm = calc_peps_norm(peps,chi=chi,singleLayer=singleLayer)
    too_high = (abs(norm) > 1.+norm_tol)
    mpiprint(-2,'Initial Norm = {}'.format(norm))

    if not too_high:
        mpiprint(-2,'Making norm greater than 1.')

    while not too_high:
        # increase tensor values
        peps = multiply_peps_elements(peps,(1.+change_int))
        # Recalculate norm
        norm = calc_peps_norm(peps,chi=chi,singleLayer=singleLayer)

        # Check if we should keep the change
        if (abs(norm) < 1.):
            # Need to increase further
            change_int *= 2.
            mpiprint(-2,'Norm ({}) below 1'.format(norm))
        else:
            mpiprint(-2,'Norm ({}) above 1.'.format(norm))
            too_high = True

    converged = False
    nIter = 0
    mpiprint(-2,'Decreasing norm towards 1')
    while not converged:
        # Check for convergence
        if (abs(norm) < 1.+norm_tol) and (abs(norm) > 1.-norm_tol):
            converged = True
            break
        else:
            nIter += 1
       
        # Try to decrease tensor values
        peps = multiply_peps_elements(peps,1./(1.+change_int))
        # Recalculate norm
        norm = calc_peps_norm(peps,chi=chi,singleLayer=singleLayer)

        # Check if we should keep the change
        if (abs(norm) < 1.):
            # We have gone too far
            peps = multiply_peps_elements(peps,(1.+change_int))
            change_int /= 2.
            mpiprint(-2,'\tNorm unchanged ({})'.format(norm))
        else:
            mpiprint(-2,'New Norm = {}'.format(norm))

    return peps

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

def make_rand_peps(Nx,Ny,d,D,dtype=float_):
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

# -----------------------------------------------------------------
# PEPS Class

class PEPS:
    """
    A class to hold and manipulate PEPS
    """

    def __init__(self,Nx=10,Ny=10,d=2,D=2,
                 chi=None,norm_tol=1e-5,
                 singleLayer=True,max_norm_iter=100,
                 norm_change_int=3e-2,dtype=float_):
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
                How close to 1. the norm should be
            singleLayer : bool
                Whether to use a single layer environment
                (currently only option implemented)
            max_norm_iter : int
                The maximum number of normalization iterations
            norm_change_int : float
                Constant to multiply peps tensor entries to 
                approach norm = 1.
            dtype : dtype
                The data type for the PEPS

        Returns:
            PEPS : PEPS Object
                The resulting random projected entangled pair
                state as a PEPS object
        """
        # Collect input arguments
        self.Nx          = Nx
        self.Ny          = Ny
        self.d           = d
        self.D           = D
        if chi is None: chi = D**2
        self.chi         = chi
        self.norm_tol    = norm_tol
        self.singleLayer = singleLayer
        self.max_norm_iter = max_norm_iter
        self.norm_change_int = norm_change_int
        self.dtype       = dtype

        # Make a random PEPS
        self.tensors = make_rand_peps(self.Nx,self.Ny,self.d,self.D,dtype=self.dtype)

        # Normalize the PEPS
        self.normalize()

    def calc_left_bound_mpo(self):
        """
        """

    def calc_right_bound_mpo(self):
        """
        """

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
        return calc_peps_norm(self.tensors,chi=chi,singleLayer=singleLayer)

    def normalize(self,max_iter=None,norm_tol=None,change_int=None,chi=None,singleLayer=None):
        """
        Normalize the full PEPS

        Args:
            self : PEPS Object
                The PEPS to be normalized

        Kwargs:
            max_iter : int
                The maximum number of iterations of the normalization 
                procedure. Default is 100.
            norm_tol : float
                How near 1. the norm must be for the normalization 
                procedure to be considered to be converged. Default
                is 100.
            change_int : float
                The magnitude with which we change entries in the 
                PEPS tensor to try to get it closer to 1. The default
                is 1e-2.
            chi : int
                The boundary MPO's maximum bond dimension, current default
                is the current bond dimension squared.
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
        if change_int is None: change_int = self.norm_change_int
        if singleLayer is None: singleLayer = self.singleLayer
        # Run the normalization procedure
        self.tensors = normalize_peps(self.tensors,
                                      max_iter = max_iter,
                                      norm_tol = norm_tol,
                                      change_int = change_int,
                                      chi = chi,
                                      singleLayer=singleLayer)
        
        # Calculate the norm
        norm = self.calc_norm(chi=chi,singleLayer=singleLayer)
        # Once complete, return norm
        return norm

