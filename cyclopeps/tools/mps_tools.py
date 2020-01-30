"""
Tools for Matrix Product States

Author: Phillip Helms <phelms@caltech.edu>
Date: July 2019

"""

from cyclopeps.tools.gen_ten import einsum
#from cyclopeps.tools.params import *
from cyclopeps.tools.utils import *
from numpy import float_
import copy
import numpy as np

def calc_site_dims(site,d,mbd,fixed_bd=False):
    """
    Calculate the dimensions for an mps tensor

    Args:
        site : int
            The current site in the mps
        d : 1D Array
            A list of the local state space dimension
            for the system
        site_d : int
            Local bond dimension at current site
        mbd : int
            Maximum retained bond dimension

    Kwargs:
        fixed_bd : bool
            If using open boundary conditions (i.e. periodic==False)
            this ensures that all bond dimensions are constant
            throughout the MPS, i.e. mps[0].dim = (1 x d[0] x mbd)
            instead of mps[0].dim = (1 x d[0] x d[0]), and so forth.

    Returns:
        dims : 1D Array
            A list of len(dims) == 3, with the dimensions of the
            tensor at the current site.
    """

    # Find lattice size
    N = len(d)

    # Current limitation: d is symmetric
    for i in range(N):
        assert d[i] == d[N-(i+1)],'Current limitation: d must be symmetric around center'

    # Find local bond dimension
    dloc = d[site]

    # First Site (special case)
    if site == 0:
        if fixed_bd:
            dims = [1,dloc,mbd]
        else:
            dims = [1,dloc,dloc]

    # Last Site (special Case)
    elif site == N-1:
        if fixed_bd:
            dims = [mbd,dloc,1]
        else:
            dims = [dloc,dloc,1]

    # Central Site (general case)
    else:
        if fixed_bd:
            dims = [mbd,dloc,mbd]
        else:
            if site < int(N/2):
                diml = min(mbd,prod((d[:site])))
                dimr = min(mbd,prod(d[:site+1]))
            elif (site == int(N/2)) and (N%2):
                diml = min(mbd,prod(d[:site]))
                dimr = min(mbd,prod(d[site+1:]))
            else:
                diml = min(mbd,prod(d[site:]))
                dimr = min(mbd,prod(d[site+1:]))
                dims = [diml,dloc,dimr]

    return dims

def calc_entanglement(S):
    """
    Calculate entanglement given a vector of singular values

    Args:
        S : 1D Array
            Singular Values

    Returns:
        EE : double
            The von neumann entanglement entropy
        EEspec : 1D Array
            The von neumann entanglement spectrum
            i.e. EEs[i] = S[i]^2*log_2(S[i]^2)
    """
    # Create a copy of S
    S = copy.deepcopy(S)
    # Ensure correct normalization
    norm_fact = sqrt(dot(S,conj(S)))
    S /= norm_fact

    # Calc Entanglement Spectrum
    EEspec = -S*conj(S)*log2(S*conj(S))

    # Sum to get Entanglement Entropy
    EE = summ(EEspec)

    # Print Results
    #mpiprint(8,'Entanglement Entropy = {}'.format(EE))
    #mpiprint(9,'Entanglement Spectrum = {}'.format(EEspec))

    # Return Results
    return EE,EEspec

def move_gauge_right_qr(ten1,ten2):
    """
    Move the gauge via qr decomposition from ten1 to ten2

    Args:
        ten1 : np or ctf array
            The site currently holding the gauge
        ten2 : np or ctf array
            The neighboring right site
    Returns:
        ten1 : np or ctf array
            The now isometrized tensor
        ten2 : np or ctf array
            The tensor now holding the gauge
    """
    # Perform the svd on the tensor
    Q,R = ten1.qr(2)
    ten1 = Q

    # Multiply remainder into neighboring site
    ten2 = einsum('ab,bpc->apc',R,ten2)

    # Return results
    return ten1,ten2

def move_gauge_left_qr(ten1,ten2):
    """
    Move the gauge via qr decomposition from ten1 to ten2

    Args:
        ten1 : np or ctf array
            The site currently holding the gauge
        ten2 : np or ctf array
            The neighboring right site
    Returns:
        ten1 : np or ctf array
            The now isometrized tensor
        ten2 : np or ctf array
            The tensor now holding the gauge
    """
    mpiprint(0,'QR to the left not implemented yet')
    raise NotImplemented

def move_gauge_right_svd(ten1,ten2,truncate_mbd=1e100):
    """
    Move the gauge via svd from ten1 to ten2

    Args:
        ten1 : np or ctf array
            The site currently holding the gauge
        ten2 : np or ctf array
            The neighboring right site

    Kwargs:
        truncate_mbd : int
            The Maximum retained Bond Dimension

    Returns:
        ten1 : np or ctf array
            The now isometrized tensor
        ten2 : np or ctf array
            The tensor now holding the gauge
        EE : float
            The von neumann entanglement entropy
            Only returned if return_ent == True
        EEs : 1D Array of floats
            The von neumann entanglement spectrum
            Only returned if return_ent == True
        wgt : float
            The sum of the discarded weigths
            Only returned if return_wgt == True
    """
    # Perform the svd on the tensor
    U,S,V,EE,EEs,wgt = ten1.svd(2,truncate_mbd=truncate_mbd)
    ten1 = U

    # Multiply remainder into neighboring site
    gauge = einsum('ab,bc->ac',S,V)
    ten2 = einsum('ab,bpc->apc',gauge,ten2)

    # Return results
    return ten1,ten2,EE,EEs,wgt

def move_gauge_left_svd(ten1,ten2,truncate_mbd=1e100):
    """
    Move the gauge via svd from ten2 to ten1

    Args:
        ten1 : np or ctf array
            The neighboring left site
        ten2 : np or ctf array
            The site currently holding the gauge

    Kwargs:
        truncate_mbd : int
            The Maximum retained Bond Dimension

    Returns:
        ten1 : np or ctf array
            The tensor now holding the gauge
        ten2 : np or ctf array
            The now isometrized tensor
        EE : float
            The von neumann entanglement entropy
            Only returned if return_ent == True
        EEs : 1D Array of floats
            The von neumann entanglement spectrum
            Only returned if return_ent == True
        wgt : float
            The sum of the discarded weigths
            Only returned if return_wgt == True
    """
    if DEBUG:
        init_cont = einsum('abc,cde->abde',ten1,ten2)
        mpiprint(0,'\t\tInitial Shape ten1 = {}'.format(ten1.ten.shape))
        mpiprint(0,'\t\tInitial Shape ten2 = {}'.format(ten2.ten.shape))
    # Perform the svd on the tensor
    U,S,V,EE,EEs,wgt = ten2.svd(1,truncate_mbd=truncate_mbd)
    ten2 = V

    # Multiply remainder into neighboring site
    gauge = einsum('ab,bc->ac',U,S)
    ten1 = einsum('apb,bc->apc',ten1,gauge)

    if DEBUG:
        fin_cont = einsum('abc,cde->abde',ten1,ten2)
        if fin_cont.sym is None:
            diff = init_cont.backend.sum(init_cont.backend.abs(init_cont.ten-fin_cont.ten))
            mpiprint(0,'\t\tFinal   Shape ten1 = {}'.format(ten1.ten.shape))
            mpiprint(0,'\t\tFinal   Shape ten2 = {}'.format(ten2.ten.shape))
            mpiprint(0,'\t\tDifference before/after SVD = {}'.format(diff))
        else:
            diff = init_cont.backend.sum(init_cont.backend.abs(init_cont.ten.array-fin_cont.ten.array))
            mpiprint(0,'\t\tFinal   Shape ten1 = {}'.format(ten1.ten.array.shape))
            mpiprint(0,'\t\tFinal   Shape ten2 = {}'.format(ten2.ten.array.shape))
            mpiprint(0,'\t\tDifference before/after SVD = {}'.format(diff))
    # Return results
    return ten1,ten2,EE,EEs,wgt

def move_gauge_right_tens(ten1,ten2,truncate_mbd=1e100,return_ent=True,return_wgt=True):
    """
    Move the gauge via either svd or qr from ten1 to ten2

    Args:
        ten1 : np or ctf array
            The site currently holding the gauge
        ten2 : np or ctf array
            The neighboring right site

    Kwargs:
        return_ent : bool
            Whether or not to return the entanglement 
            entropy and entanglement spectrum
            Default: True
        return_wgt : bool
            Whether or not to return the sum of the 
            discarded weights.
            Default: True
        truncate_mbd : int
            The Maximum retained Bond Dimension

    Returns:
        ten1 : np or ctf array
            The now isometrized tensor
        ten2 : np or ctf array
            The tensor now holding the gauge
        EE : float
            The von neumann entanglement entropy
            Only returned if return_ent == True
            and the bond dimension is smaller than 
            truncate_mbd
        EEs : 1D Array of floats
            The von neumann entanglement spectrum
            Only returned if return_ent == True
            and the bond dimension is smaller than 
            truncate_mbd
        wgt : float
            The sum of the discarded weigths
            Only returned if return_wgt == True
            and the bond dimension is smaller than 
            truncate_mbd
    """
    mpiprint(9,'Moving loaded tensors gauge right')
    
    # Do either svd or qr decomposition
    if truncate_mbd > ten1.shape[2]:
        ten1,ten2 = move_gauge_right_qr(ten1,ten2)
        EE,EEs,wgt = None,None,None
        #ten1,ten2,EE,EEs,wgt = move_gauge_right_svd(ten1,ten2)
    else:
        ten1,ten2,EE,EEs,wgt = move_gauge_right_svd(ten1,ten2,truncate_mbd=truncate_mbd)

    # Return Results
    if return_wgt and return_ent:
        return ten1,ten2,EE,EEs,wgt
    elif return_wgt:
        return ten1,ten2,wgt
    elif return_ent:
        return ten1,ten2,EE,EEs
    else:
        return ten1,ten2

def move_gauge_left_tens(ten1,ten2,truncate_mbd=1e100,return_ent=True,return_wgt=True):
    """
    Move the gauge via svd from ten2 to ten1

    Args:
        ten1 : np or ctf array
            The neighboring left site
        ten2 : np or ctf array
            The site currently holding the gauge

    Kwargs:
        return_ent : bool
            Whether or not to return the entanglement 
            entropy and entanglement spectrum
            Default: True
        return_wgt : bool
            Whether or not to return the sum of the 
            discarded weights.
            Default: True
        truncate_mbd : int
            The Maximum retained Bond Dimension

    Returns:
        ten1 : np or ctf array
            The tensor now holding the gauge
        ten2 : np or ctf array
            The now isometrized tensor
        EE : float
            The von neumann entanglement entropy
            Only returned if return_ent == True
        EEs : 1D Array of floats
            The von neumann entanglement spectrum
            Only returned if return_ent == True
        wgt : float
            The sum of the discarded weigths
            Only returned if return_wgt == True
    """
    mpiprint(9,'Moving loaded tensors gauge right')
    
    # Do either svd or qr decomposition
    if False: # NOTE - Figure out how to do this correctly #truncate_mbd > ten1.shape[2]:
        #ten1,ten2 = move_gauge_left_qr(ten1,ten2)
        ten1,ten2,_,_,_ = move_gauge_left_svd(ten1,ten2)
        EE,EEs,wgt = None,None,None
    else:
        ten1,ten2,EE,EEs,wgt = move_gauge_left_svd(ten1,ten2,truncate_mbd=truncate_mbd)

    # Return Results
    if return_wgt and return_ent:
        return ten1,ten2,EE,EEs,wgt
    elif return_wgt:
        return ten1,ten2,wgt
    elif return_ent:
        return ten1,ten2,EE,EEs
    else:
        return ten1,ten2

def move_gauge_right(mps,site,truncate_mbd=1e100,return_ent=True,return_wgt=True):
    """
    Move the gauge via svd from ten1 to ten2

    Args:
        ten1 : np or ctf array
            The site currently holding the gauge
        ten2 : np or ctf array
            The neighboring right site

    Kwargs:
        return_ent : bool
            Whether or not to return the entanglement 
            entropy and entanglement spectrum
            Default: True
        return_wgt : bool
            Whether or not to return the sum of the 
            discarded weights.
            Default: True
        truncate_mbd : int
            The Maximum retained Bond Dimension

    Returns:
        ten1 : np or ctf array
            The now isometrized tensor
        ten2 : np or ctf array
            The tensor now holding the gauge
        EE : float
            The von neumann entanglement entropy
            Only returned if return_ent == True
        EEs : 1D Array of floats
            The von neumann entanglement spectrum
            Only returned if return_ent == True
        wgt : float
            The sum of the discarded weigths
            Only returned if return_wgt == True
    """
    # Retrieve the relevant tensors
    ten1 = mps[site]
    ten2 = mps[site+1]

    # Check to make sure everything is done correctly
    if DEBUG:
        init_cont = einsum('abc,cde->abde',mps[site],mps[site+1])
        if ten1.sym is None:
            mpiprint(0,'\tInitial Shape ten1 = {}, {}'.format(mps[site].ten.shape,mps[site].legs))
            mpiprint(0,'\tInitial Shape ten2 = {}, {}'.format(mps[site+1].ten.shape,mps[site+1].legs))
        else:
            mpiprint(0,'\tInitial Shape ten1 = {}, {}, {}, {}'.format(mps[site].ten.array.shape,mps[site].legs,mps[site].sym[0],mps[site].sym[1]))
            mpiprint(0,'\tInitial Shape ten2 = {}, {}, {}, {}'.format(mps[site+1].ten.array.shape,mps[site+1].legs,mps[site+1].sym[0],mps[site+1].sym[1]))

    # Move the gauge
    ten1,ten2,EE,EEs,wgt = move_gauge_right_tens(ten1,ten2,
                                                truncate_mbd=truncate_mbd,
                                                return_ent=True,
                                                return_wgt=True)

    # Put back into the mps
    mps[site] = ten1
    mps[site+1] = ten2

    if DEBUG:
        fin_cont = einsum('abc,cde->abde',mps[site],mps[site+1])
        if fin_cont.sym is None:
            diff = init_cont.backend.sum(init_cont.backend.abs(init_cont.ten-fin_cont.ten))
            mpiprint(0,'\tFinal   Shape ten1 = {}, {}'.format(mps[site].ten.shape,mps[site].legs))
            mpiprint(0,'\tFinal   Shape ten2 = {}, {}'.format(mps[site+1].ten.shape,mps[site+1].legs))
            mpiprint(0,'\tDifference before/after QR/SVD = {}'.format(diff))
        else:
            diff = init_cont.backend.sum(init_cont.backend.abs(init_cont.ten.make_sparse()-fin_cont.ten.make_sparse()))
            mpiprint(0,'\tFinal   Shape ten1 = {}, {}, {}, {}'.format(mps[site].ten.array.shape,mps[site].legs,mps[site].sym[0],mps[site].sym[1]))
            mpiprint(0,'\tFinal   Shape ten2 = {}, {}, {}, {}'.format(mps[site+1].ten.array.shape,mps[site+1].legs,mps[site+1].sym[0],mps[site+1].sym[1]))
            mpiprint(0,'\tDifference before/after QR/SVD = {}'.format(diff))

    # Return results
    if return_wgt and return_ent:
        return mps,EE,EEs,wgt
    elif return_wgt:
        return mps,wgt
    elif return_ent:
        return mps,EE,EEs
    else:
        return mps

def move_gauge_left(mps,site,truncate_mbd=1e100,return_ent=True,return_wgt=True):
    """
    Move the gauge via svd from ten2 to ten1

    Args:
        ten1 : np or ctf array
            The neighboring left site
        ten2 : np or ctf array
            The site currently holding the gauge

    Kwargs:
        return_ent : bool
            Whether or not to return the entanglement 
            entropy and entanglement spectrum
            Default: True
        return_wgt : bool
            Whether or not to return the sum of the 
            discarded weights.
            Default: True
        truncate_mbd : int
            The Maximum retained Bond Dimension

    Returns:
        ten1 : np or ctf array
            The tensor now holding the gauge
        ten2 : np or ctf array
            The now isometrized tensor
        EE : float
            The von neumann entanglement entropy
            Only returned if return_ent == True
        EEs : 1D Array of floats
            The von neumann entanglement spectrum
            Only returned if return_ent == True
        wgt : float
            The sum of the discarded weigths
            Only returned if return_wgt == True
    """
    # Retrieve the relevant tensors
    ten1 = mps[site-1]
    ten2 = mps[site]

    if DEBUG:
        init_cont = einsum('abc,cde->abde',mps[site-1],mps[site])
        if ten1.sym is None:
            mpiprint(0,'\tInitial Shape ten1 = {}, {}'.format(mps[site-1].ten.shape,mps[site-1].legs))
            mpiprint(0,'\tInitial Shape ten2 = {}, {}'.format(mps[site].ten.shape,mps[site].legs))
        else:
            mpiprint(0,'\tInitial Shape ten1 = {}, {}, {}, {}'.format(mps[site-1].ten.array.shape,mps[site-1].legs,mps[site-1].sym[0],mps[site-1].sym[1]))
            mpiprint(0,'\tInitial Shape ten2 = {}, {}, {}, {}'.format(mps[site].ten.array.shape,mps[site].legs,mps[site].sym[0],mps[site].sym[1]))

    # Move the gauge
    ten1,ten2,EE,EEs,wgt = move_gauge_left_tens(ten1,ten2,
                                                truncate_mbd=truncate_mbd,
                                                return_ent=True,
                                                return_wgt=True)

    # Put back into the mps
    mps[site-1] = ten1
    mps[site] = ten2

    if DEBUG:
        fin_cont = einsum('abc,cde->abde',mps[site-1],mps[site])
        if fin_cont.sym is None:
            diff = init_cont.backend.sum(init_cont.backend.abs(init_cont.ten-fin_cont.ten))
            mpiprint(0,'\tFinal   Shape ten1 = {}, {}'.format(mps[site-1].ten.shape,mps[site-1].legs))
            mpiprint(0,'\tFinal   Shape ten2 = {}, {}'.format(mps[site].ten.shape,mps[site].legs))
            mpiprint(0,'\tDifference before/after QR/SVD = {}'.format(diff))
        else:
            diff = init_cont.backend.sum(init_cont.backend.abs(init_cont.ten.make_sparse()-fin_cont.ten.make_sparse()))
            mpiprint(0,'\tFinal   Shape ten1 = {}, {}, {}, {}'.format(mps[site-1].ten.array.shape,mps[site-1].legs,mps[site-1].sym[0],mps[site-1].sym[1]))
            mpiprint(0,'\tFinal   Shape ten2 = {}, {}, {}, {}'.format(mps[site].ten.array.shape,mps[site].legs,mps[site].sym[0],mps[site].sym[1]))
            mpiprint(0,'\tDifference before/after QR/SVD = {}'.format(diff))

    # Return results
    if return_wgt and return_ent:
        return mps,EE,EEs,wgt
    elif return_wgt:
        return mps,wgt
    elif return_ent:
        return mps,EE,EEs
    else:
        return mps

def make_mps_left(mps,truncate_mbd=1e100):
    """
    Put an mps into left canonical form

    Args:
        mps : list of mps tensors
            The MPS stored as a list of mps tensors
    
    Kwargs:
        truncate_mbd : int
            The maximum bond dimension to which the 
            mps should be truncated

    Returns:
        mps : list of mps tensors
            The resulting left-canonicalized MPS
    """
    # Figure out size of mps
    N = len(mps)
    # Loop backwards
    for site in range(N-1):
        mps = move_gauge_right(mps,site,
                               truncate_mbd=truncate_mbd,
                               return_ent=False,
                               return_wgt=False)

    # Return results
    return mps

def make_mps_right(mps,truncate_mbd=1e100):
    """
    Put an mps into right canonical form

    Args:
        mps : list of mps tensors
            The MPS stored as a list of mps tensors
    
    Kwargs:
        truncate_mbd : int
            The maximum bond dimension to which the 
            mps should be truncated

    Returns:
        mps : list of mps tensors
            The resulting right-canonicalized MPS
    """
    # Figure out size of mps
    N = len(mps)
    # Loop backwards
    for site in range(int(N)-1,0,-1):
        mps = move_gauge_left(mps,site,
                              truncate_mbd=truncate_mbd,
                              return_ent=False,
                              return_wgt=False)

    # Return results
    return mps

def mps_apply_svd(mps,chi):
    """
    Shrink the maximum bond dimension of an mps

    Args:
        mps : List of mps tensors
            The mps which will be used
        chi : int
            The new maximum bond dimension

    Returns:
        mps : List of mps tensors
            The mps with a maximum bond dimension of \chi
    """
    mpiprint(8,'Moving gauge to left, prep for truncation')
    mps = make_mps_left(mps)
    mpiprint(8,'Truncating as moving to right')
    mps = make_mps_right(mps,truncate_mbd=chi)
    return mps

def identity_mps(N,dtype=float_):
    """
    Create an identity mps (currently limited to physical and 
    auxilliary bond dimension = 1).
    """
    mps = MPS(d=1,D=1,N=N,dtype=dtype)
    for site in range(N):
        mps[site][:,:,:] = 1.
    return mps

# -----------------------------------------------------------------
# MPS Class
class MPS:
    """
    A class to hold and manipulate matrix product states.
    """
    def __init__(self,tens):
        """
        Create a random MPS object

        Args:
            self : MPS Object
            tens : list
                A list of the tensors (as gen_ten objects)
                to put into the mps

        Returns:
            MPS : MPS Object
                The resulting Matrix Product State stored as 
                an MPS Object

        .. To Do:
            * Store tensors on disk
        """
        # Collect input arguments
        self.N = len(tens)

        # Create a list to hold all tensors
        self.tensors = [None]*self.N

        # Update tensors
        for site in range(len(tens)):

            # Copy Tensor
            self[site] = tens[site].copy()

    def copy(self):
        """
        Returns a copy of the current MPS
        """
        return MPS([self[site].copy() for site in range(self.N)])

    def __getitem__(self,i):
        """
        Returns the MPS tensor at site i
        """
        if not hasattr(i,'__len__'):
            return self.tensors[i]
        else:
            return [self.tensors[i[ind]] for ind in range(len(i))]

    def __setitem__(self, site, ten):
        # Update tensor
        self.tensors[site] = ten

    def __mult__(self, const):
        """
        Multiply an MPS by a constant
        """
        mps_tmp = self.copy()
        for site in range(self.N):
            mps_tmp[site] = mps_tmp[site] * (abs(const)**(1./mps_tmp.N))
            if (const < 0.) and (site == 0):
                mps_tmp[site] *= -1.
        return mps_tmp

    def norm(self):
        """
        Compute the norm of the MPS
        """
        # Move to the left, contracting env with bra and ket tensors
        for site in range(self.N):
            if site == 0:
                # Form initial norm environment
                norm_env = einsum('apb,ApB->aAbB',self[site],self[site].conj())
                # Remove initial empty indices
                norm_env = norm_env.remove_empty_ind(0)
                norm_env = norm_env.remove_empty_ind(0)
            else:
                # Add next mps tensors to norm environment
                tmp1 = einsum('aA,apb->Apb',norm_env,self[site])
                norm_env = einsum('Apb,ApB->bB',tmp1,self[site].conj())
        # Extract and return result
        norm = norm_env.to_val()
        return norm

    def apply_svd(self,chi):
        """
        Shrink the maximum bond dimension of an mps

        Args:
            self : MPS Object
                The mps which will be used

        Kwargs:
            chi : int
                The new maximum bond dimension
        """
        res = self.copy()
        res = mps_apply_svd(res.tensors,chi)
        return MPS(res)

    def make_left_canonical(self,chi=1e100):
        """
        Make the MPS Left canonical (and possibly truncate bond dim)

        Args:
            self : MPS Object
                The mps which will be used

        Kwargs:
            chi : int
                The new maximum bond dimension
        """
        res = self.copy()
        res = make_mps_left(res.tensors,truncate_mbd=chi)
        return MPS(res)

    def make_right_canonical(self,chi=1e100):
        """
        Make the MPS Right canonical (and possibly truncate bond dim)

        Args:
            self : MPS Object
                The mps which will be used

        Kwargs:
            chi : int
                The new maximum bond dimension
        """
        res = self.copy()
        res = make_mps_right(res.tensors,truncate_mbd=chi)
        return MPS(res)

    def move_gauge_right(self,site,truncate_mbd=1e100):
        """
        Do a tensor decomposition on site to move the gauge to site+1

        Args:
            self : MPS Object
                The mps which will be used
            site : int
                The starting location of the MPS gauge

        Kwargs:
            truncate_mbd : int
                The bond dimension to be truncated to
        """
        res = self.copy()
        res = move_gauge_right(res.tensors,site,
                               truncate_mbd=truncate_mbd,
                               return_ent=False,
                               return_wgt=False)
        return MPS(res)

    def move_gauge_left(self,site,truncate_mbd=1e100):
        """
        Do a tensor decomposition on site to move the gauge to site-1

        Args:
            self : MPS Object
                The mps which will be used
            site : int
                The starting location of the MPS gauge

        Kwargs:
            truncate_mbd : int
                The bond dimension to be truncated to
        """
        res = self.copy()
        res = move_gauge_left(res.tensors,site,
                               truncate_mbd=truncate_mbd,
                               return_ent=False,
                               return_wgt=False)
        return MPS(res)

    def input_mps_list(self,new):
        """
        Take a list and put it into our MPS
        """
        # Update tensors
        for site in range(len(new)):

            # Copy Tensor
            self[site] = new[site].copy()

    def contract(self,mps2):
        """
        Contract an mps with another mps

        Args:
            self : MPS Object
                The mps which will be contracted
            mps2 : MPS Object
                The second mps which will be contracted

        Returns:
            res : float
                The resulting scalar from the contraction
        """
        # Move to the left, contracting env with bra and ket tensors
        for site in range(self.N):
            if site == 0:
                # Form initial norm environment
                norm_env = einsum('apb,ApB->aAbB',self[site],mps2[site])
                # Remove initial empty indices
                norm_env = norm_env.remove_empty_ind(0)
                norm_env = norm_env.remove_empty_ind(0)
            else:
                # Add next mps tensors to norm environment
                tmp1 = einsum('aA,apb->Apb',norm_env,self[site])
                norm_env = einsum('Apb,ApB->bB',tmp1,mps2[site])
        # Extract and return result
        norm = norm_env.to_val()
        return norm

    def norm(self):
        """
        Compute the norm of the MPS
        """
        return self.contract(self.conj())

    def __len__(self):
        return len(self.tensors)

    def max_elem(self,absolute=True):
        try: 
            backend = self[0].backend
        except:
            backend = np
        maxval = 0
        for i in range(len(self)):
            if absolute:
                maxval = max(self.tensors[i].abs().max(),maxval)
            else:
                maxval = max(self.tensors[i].max(),maxval)
        return maxval

    def conj(self):
        mpsconj = self.copy()
        for site in range(len(mpsconj.tensors)):
            mpsconj[site] = mpsconj[site].conj()
        return mpsconj

    # -----------------------------------------------------------------------
    # Yet to be implemented functions
    #def __add__(self, mps):
    #def __sub__(self,mps):
    #    return self+mps*(-1.)
    #def fidel(self,mps):
    #def product(self,mps):
    #def distance(self,mps):
    #def normalize(self):
    #def appSVD(self,chi):
    #def appINV(self,chi):
