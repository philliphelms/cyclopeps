"""
Tools for Matrix Product States

Author: Phillip Helms <phelms@caltech.edu>
Date: July 2019

"""

from cyclopeps.tools.params import *
from cyclopeps.tools.utils import *
from numpy import float_
import copy

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

def qr_ten(ten,split_ind,rq=False):
    """
    Compute the QR Decomposition of an input tensor

    Args:
        ten : ctf or np array
            Array for which the svd will be done
        split_ind : int
            The dimension where the split into a matrix will be made

    Kwargs:
        rq : bool
            If True, then RQ decomposition will be done
            instead of QR decomposition

    Returns:
        Q : ctf or np array
            The resulting Q matrix 
        R : ctf or np array
            The resulting R matrix
    """
    mpiprint(9,'Performing qr on tensors')
    # Reshape tensor into matrix
    ten_shape = ten.shape
    mpiprint(9,'First, reshape the tensor into a matrix')
    ten = ten.reshape([prod(ten_shape[:split_ind]),prod(ten_shape[split_ind:])])

    # Perform svd
    mpiprint(9,'Perform actual qr')
    if not rq:
        # Do the QR Decomposition
        Q,R = qr(ten)

        # Reshape to match correct tensor format
        mpiprint(9,'Reshape to match original tensor dimensions')
        new_dims = ten_shape[:split_ind]+(prod(Q.shape)/prod(ten_shape[:split_ind]),)
        Q = Q.reshape(new_dims)
        new_dims = (prod(R.shape)/prod(ten_shape[split_ind:]),)+ten_shape[split_ind:]
        R = R.reshape(new_dims)

        # Quick check to see if it worked
        # assert(np.allclose(ten,np.reshape(np.dot(Q,R),ten.shape),rtol=1e-6))
    else:
        # Do the QR decomposition
        ten_ = copy.deepcopy(ten[::-1,:])
        Q_,R_ = qr(transpose(ten_))
        Q = Q_[::-1,:]
        R = transpose(R_)[::-1,::-1]

        # Reshape to match correct tensor format
        mpiprint(9,'Reshape to match original tensor dimensions')
        new_dims = ten_shape[:split_ind]+(prod(R.shape)/prod(ten_shape[:split_ind]),)
        R = R.reshape(new_dims)
        new_dims = (prod(Q.shape)/prod(ten_shape[split_ind:]),)+ten_shape[split_ind:]
        Q = Q.reshape(new_dims)

        # Quick check to see if it worked...
        # assert(np.allclose(ten,np.reshape(np.einsum('ij,jkl->ikl',R,Q),ten.shape),rtol=1e-6))

    # Return results
    if rq:
        return R,Q
    else:
        return Q,R

def svd_ten(ten,split_ind,truncate_mbd=1e100,return_ent=True,return_wgt=True):
    """
    Compute the Singular Value Decomposition of an input tensor

    Args:
        ten : ctf or np array
            Array for which the svd will be done
        split_ind : int
            The dimension where the split into a matrix will be made

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
        U : ctf or np array
            The resulting U matrix from the svd
        S : ctf or np array
            The resulting singular values from the svd
        V : ctf or np array
            The resulting V matrix from the svd
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
    mpiprint(8,'Performing svd on tensors')
    # Reshape tensor into matrix
    ten_shape = ten.shape
    mpiprint(9,'First, reshape the tensor into a matrix')
    ten = ten.reshape([prod(ten_shape[:split_ind]),prod(ten_shape[split_ind:])])

    # Perform svd
    mpiprint(9,'Perform actual svd')
    U,S,V = svd(ten)

    # Compute Entanglement
    mpiprint(9,'Calculate the entanglment')
    EE,EEs = calc_entanglement(S)

    # Truncate results (if necessary)
    D = S.shape[0]

    # Make sure D is not larger than allowed
    if truncate_mbd is not None:
        D = int(min(D,truncate_mbd))

    # Compute amount of S discarded
    wgt = S[D:].sum()

    # Truncate U,S,V
    mpiprint(9,'Limit tensor dimensions')
    U = U[:,:D]
    S = S[:D]
    V = V[:D,:]

    # Reshape to match correct tensor format
    mpiprint(9,'Reshape to match original tensor dimensions')
    new_dims = ten_shape[:split_ind]+(prod(U.shape)/prod(ten_shape[:split_ind]),)
    U = U.reshape(new_dims)
    new_dims = (prod(V.shape)/prod(ten_shape[split_ind:]),)+ten_shape[split_ind:]
    V = V.reshape(new_dims)

    # Print some results
    mpiprint(10,'Entanglement Entropy = {}'.format(EE))
    mpiprint(12,'EE Spectrum = ')
    nEEs = EEs.shape[0]
    for i in range(nEEs):
        mpiprint(12,'   {}'.format(EEs[i]))
    mpiprint(11,'Discarded weights = {}'.format(wgt))

    # Return results
    if return_wgt and return_ent:
        return U,S,V,EE,EEs,wgt
    elif return_wgt:
        return U,S,V,wgt
    elif return_ent:
        return U,S,V,EE,EEs
    else:
        return U,S,V

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
    # Find tensor dimensions
    (n1,n2,n3) = ten1.shape
    (n4,n5,n6) = ten2.shape

    # Perform the svd on the tensor
    Q,R = qr_ten(ten1,2)

    # Pad result to keep bond dim fixed
    mpiprint(8,'Padding tensors from svd')
    Qpad = zeros((n1,n2,n3),dtype=type(Q[0,0,0]))
    Qpad[:,:,:min(n1*n2,n3)] = Q
    Rpad = zeros((n3,n4),dtype=type(R[0,0]))
    Rpad[:min(n1*n2,n3),:] = R
    ten1 = Qpad

    # Multiply remainder into neighboring site
    mpiprint(8,'Actually moving the gauge')
    ten2 = einsum('ab,bpc->apc',Rpad,ten2)

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
    # Find tensor dimensions
    (n1,n2,n3) = ten1.shape
    (n4,n5,n6) = ten2.shape

    # Perform the svd on the tensor
    R,Q = qr_ten(ten2,1,rq=True)

    # Pad result to keep bond dim fixed
    Qpad = zeros((n4,n5,n6),dtype=type(Q[0,0,0]))
    Qpad[:min(n4,n5*n6),:,:] = Q
    Rpad = zeros((n3,n4),dtype=type(R[0,0]))
    Rpad[:,:min(n4,n5*n6)] = R
    ten2 = Qpad

    # Multiply remainder ininto neighboring site
    mpiprint(8,'Actually moving the gauge')
    ten1 = einsum('apb,bc->apc',ten1,Rpad)

    # Return results
    return ten1,ten2

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
    # Find tensor dimensions
    (n1,n2,n3) = ten1.shape
    (n4,n5,n6) = ten2.shape

    # Perform the svd on the tensor
    U,S,V,EE,EEs,wgt = svd_ten(ten1,2,truncate_mbd=truncate_mbd)

    # Pad result to keep bond dim fixed
    mpiprint(8,'Padding tensors from svd')
    Upad = zeros((n1,n2,min(n3,truncate_mbd)),dtype=type(U[0,0,0]))
    Upad[:,:,:min(n1*n2,n3,truncate_mbd)] = U
    Spad = zeros((min(n3,truncate_mbd)),dtype=type(S[0]))
    Spad[:min(n1*n2,n3,truncate_mbd)] = S
    Vpad = zeros((min(n3,truncate_mbd),n4),dtype=type(V[0,0]))
    Vpad[:min(n1*n2,n3,truncate_mbd),:] = V
    ten1 = Upad

    # Multiply remainder into neighboring site
    mpiprint(8,'Actually moving the gauge')
    gauge = einsum('a,ab->ab',Spad,Vpad)
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
    (n1,n2,n3) = ten1.shape
    (n4,n5,n6) = ten2.shape

    # Perform the svd on the tensor
    U,S,V,EE,EEs,wgt = svd_ten(ten2,1,truncate_mbd=truncate_mbd)

    # Pad result to keep bond dim fixed
    mpiprint(8,'Padding tensors from svd')
    Vpad = zeros((min(n4,truncate_mbd),n5,n6),dtype=type(V[0,0,0]))
    Vpad[:min(n4,n5*n6,truncate_mbd),:,:] = V
    Spad = zeros((min(n4,truncate_mbd)),dtype=type(S[0]))
    Spad[:min(n4,n5*n6,truncate_mbd)] = S
    Upad = zeros((n3,min(n4,truncate_mbd)),dtype=type(U[0,0]))
    Upad[:,:min(n4,n5*n6,truncate_mbd)] = U
    ten2 = Vpad

    # Multiply remainder into neighboring site
    mpiprint(8,'Actually moving the gauge')
    gauge = einsum('ab,b->ab',Upad,Spad)
    ten1 = einsum('apb,bc->apc',ten1,gauge)
    
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
    if False: # PH - This is not working!!! #truncate_mbd > ten1.shape[2]:
        ten1,ten2 = move_gauge_left_qr(ten1,ten2)
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

    # Move the gauge
    ten1,ten2,EE,EEs,wgt = move_gauge_right_tens(ten1,ten2,
                                                truncate_mbd=truncate_mbd,
                                                return_ent=True,
                                                return_wgt=True)

    # Put back into the mps
    mps[site] = ten1
    mps[site+1] = ten2

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

    # Move the gauge
    ten1,ten2,EE,EEs,wgt = move_gauge_left_tens(ten1,ten2,
                                                truncate_mbd=truncate_mbd,
                                                return_ent=True,
                                                return_wgt=True)

    # Put back into the mps
    mps[site-1] = ten1
    mps[site] = ten2

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

def alloc_mps_env(mps,mpoL,dtype=float_):
    """ 
    Allocate tensors for the mps|mpo|mps sandwich environment tensors

    Args:
        mps : 1D List of MPS tensors
            The matrix product state
        mpoL : List of list of MPO tensors
            A list of mpos

    Returns:
        envL : 1D Array
            An environment
    """
    mpiprint(9,'Allocating mps environment')

    # Get details about mpo and mps
    nSite  = len(mps)
    nOps   = len(mpoL)

    # Initialize empty list to hold envL
    envL = []
    
    # Loop over all operators in mpo List
    for op in range(nOps):
        # Create an environment (in a list) for given operator
        env = []

        # Initial entry (for edge) PH - Does not work for periodic
        dims = (1,1,1)
        ten = zeros(dims,dtype=dtype)
        ten[0,0,0] = 1.
        # Add to env list
        env.append(ten)
        
        # Central Entries 
        for site in range(nSite-1):
            # Find required dimensions
            mps_D = mps[site].shape[2]
            for op in range(nOps):
                mpo_D=1
                if mpoL[op][site] is not None:
                    _,_,_,mpo_D = mpoL[op][site].shape
            # Create tensor
            dims = (mps_D,mpo_D,mps_D)
            ten = zeros(dims,dtype=dtype)
            # Add to env list
            env.append(ten)

        # Final entry (for right edge) PH - Does not work for periodic
        dims = (1,1,1)
        ten = zeros(dims,dtype=dtype)
        ten[0,0,0] = 1.
        # Add to env list
        env.append(ten)

        # Add new env to the overall env list
        envL.append(env)

    # Return list of environments
    return envL

def update_env_left(mpsList,mpoList,envList,site,mpslList=None):
    """
    Update the environment tensors for an mps|mpo|mps sandwich
    as we move to the left

    Args:
        mpsList : 1D Array of Matrix Product States
            A list containing a matrix product 
            states        
        mpoList : 1D Array of MPOs
            A list containing multiple MPOs
        envList : 1D Array of an environment
            A list containing multiple environments,
        site : int
            The current site

    Kwargs:
        mpslList : 1D Array of Matrix Product State
            An optional mpsList representing the left state of the 
            system if you would like different left and right vectors
            used in computing the environment

    Returns:
        envList : 1D Array of an environment
            This returns the same environmnet you started with, though
            the entries have been updated
    """
    mpiprint(8,'\tUpdating Environment moving from site {} to {}'.format(site,site-1))

    # Get useful info
    nOp = len(mpoList)
    nSite = len(mpsList)

    # Get a left MPS, 
    if mpslList is None: mpslList = mpsList
    
    # Loop through all operators
    for op in range(nOp):
        # Load environment and mps
        envLoc = envList[op][site+1]
        mpsLoc = mpsList[site]
        mpslLoc= mpslList[site]

        # Do Contraction to update env
        if mpoList[op][site] is None:
            envNew = einsum('apb,Bcb->apcB',mpsLoc,envLoc)
            envNew = einsum('apcB,ApB->Aca',envNew,conj(mpslLoc))
        else:
            envNew = einsum('apb,Bcb->apcB',mpsLoc,envLoc)
            envNew = einsum('apcB,dqpc->adqB',envNew,mpoList[op][site]) # PH - Might be dpqc?
            envNew = einsum('adqB,AqB->Ada',envNew,conj(mpslLoc))

        # Save resulting new environment
        envList[op][site] = envNew
    
    # Return environment list
    return envList

def update_env_right(mpsList,mpoList,envList,site,mpslList=None):
    """
    Update the environment tensors for an mps|mpo|mps sandwich
    as we move to the right

    Args:
        mpsList : 1D Array of Matrix Product States
            A list containing multiple matrix product 
            states 
        mpoList : 1D Array of MPOs
            A list containing multiple MPOs
        envList : 1D Array of an environment
            A list containing multiple environments,
        site : int
            The current site

    Kwargs:
        mpslList : 1D Array of Matrix Product State
            An optional mpsList representing the left state of the 
            system if you would like left and right vectors
            used in computing the environment

    Returns:
        envList : 1D Array of an environment
            This returns the same environmnet you started with, though
            the entries have been updated
    """
    mpiprint(8,'\tUpdating Environment moving from site {} to {}'.format(site,site+1))

    # Get useful info
    nOp = len(mpoList)
    nSite = len(mpsList)

    # Get a left MPS, 
    if mpslList is None: mpslList = mpsList
    
    # Loop through all operators
    for op in range(nOp):
        # Load environment and mps
        envLoc = envList[op][site]
        mpsLoc = mpsList[site]
        mpslLoc= mpslList[site]

        # Do Contraction to update env
        if mpoList[op][site] is None:
            envNew = einsum('Ala,APB->alPB',envLoc,conj(mpslLoc))
            envNew = einsum('aPb,alPB->Blb',mpsLoc,envNew)
        else:
            envNew = einsum('Ala,APB->alPB',envLoc,conj(mpslLoc))
            envNew = einsum('lPpr,alPB->aprB',mpoList[op][site],envNew)
            envNew = einsum('apb,aprB->Brb',mpsLoc,envNew)

        # Save resulting new environment
        envList[op][site+1] = envNew
    
    # Return environment list
    return envList

def calc_mps_env(mpsList,mpoList,mpslList=None,dtype=None,gSite=0,state=0):
    """
    Calculate all the environment tensors for an mps|mpo|mps sandwich

    Args:
        mpsList : 1D Array of Matrix Product States
            A list containing multiple matrix product 
            states, each of which is a list of a dictionary,
            which contains the filename and dimensions of 
            the local MPS tensor. 
        mpoList : 1D Array of MPOs
            A list containing multiple MPOs

    Kwargs:
        mpslList : 1D Array of Matrix Product State
            An optional mpsList representing the left state of the 
            system if you would like left and right vectors
            used in computing the environment
        dtype : dtype
            Specify the data type, if None, then this will use the 
            same datatype as the mpsList.
        gSite : int
            The site where the gauge is currently located. The default
            is gSite=0, meaning the mps is right canonical.

    Returns:
        envList : 1D Array of an environment
            This returns the same environmnet you started with, though
            the entries have been updated
    """
    mpiprint(8,'Calculating full mps environment')
    nSite = len(mpsList)
    # Make env same dtype as mps
    if dtype is None: dtype = mpsList[0].dtype
    # Allocate Environment
    envList = alloc_mps_env(mpsList,mpoList,dtype=dtype)
    # Calculate Environment from the right
    for site in range(nSite-1,gSite,-1):
        envList = update_env_left(mpsList,mpoList,envList,site,mpslList=mpslList)
    # Calculate Environment from the left
    for site in range(gSite):
        envList = update_env_right(mpsList,mpoList,envList,site,mpslList=mpslList)
    return envList

def contract_mps(mps1,mps2=None,mpo=None):
    """
    Contract 2 MPSs to give scalar product

    Args:
        mps1 : List of mps tensors
            The first MPS to be contracted

    Kwargs:
        mps2 : List of mps tensors
            The second MPS to be contracted. If None
            then will be set as conj(mps1)
        mpo : List of mpo tensors
            The mpo to be sandwhiched between
            the two mps. If None, then the mpo
            will be set as identities. 

    Returns:
        res : float
            The resulting value of the contraction
    """
    # Figure out mps size
    N = len(mps1)

    # Make MPO if needed
    if mpo is None:
        mpo = [[None]*N]

    # Calculate mps env from right
    env = calc_mps_env(mps1,mpo,mpslList=mps2,gSite=-1)

    # Extract and Return Result
    nOps = len(mpo)
    res = 0.
    for opind in range(nOps):
        res += env[opind][0][0,0,0]
    return res

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
    def __init__(self, d=2, D=2, N=10, fixed_bd=True, dtype=float_):
        """
        Create a random MPS object

        Args:
            self : MPS Object

        Kwargs:
            d : int
                The physical bond dimension
            D : int 
                The maximum auxilliary bond dimension
            N : int
                The number of tensors
            fixed_bd : bool
                ensures that all bond dimensions are constant
                throughout the MPS, i.e. mps[0].dim = (1 x d[0] x mbd)
                instead of mps[0].dim = (1 x d[0] x d[0]), and so forth.
            dtype : dtype
                The datatype for the tensors. Default is np.float_

        Returns:
            MPS : MPS Object
                The resulting Matrix Product State stored as 
                an MPS Object

        .. To Do:
            * Store tensors on disk
        """
        # Collect input arguments
        self.N = N
        if isinstance(d,list):
            self.d = d
        else:
            self.d = [d]*N
        self.D = D
        self.dtype = dtype

        # Determine the MPS Shape
        self.ten_dims = [None]*N
        for site in range(self.N):
            self.ten_dims[site] = calc_site_dims(site,self.d,self.D,fixed_bd=fixed_bd)

        # Create a list to hold all tensors
        self.tensors = [None]*N

        # Generate Random MPS
        for site in range(self.N):
            self.tensors[site] = rand(self.ten_dims[site],dtype=self.dtype)

    def input_mps_list(self,mps_list):
        """
        Put an mps currently stored as a list into the MPS object
        Args:
            mps_list : 1D array of uni10 tensors
                The MPS stored as a 1D array of uni10 tensors
        """
        # Change length of mps
        self.N = len(mps_list)
        self.tensors = [None]*self.N
        self.ten_dims = [None]*self.N
        self.d = [None]*self.N

        # Determine new dtype
        self.dtype = mps_list[0].dtype
        
        # Update tensors
        for site in range(len(mps_list)):

            # Copy Tensor
            self[site] = copy.deepcopy(mps_list[site])

            # Update tensor shape
            (DL,d,DR) = self[site].shape
            self.ten_dims[site] = [DL,d,DR]

            # Update local bond dimension
            self.d[site] = d

    def copy(self):
        """
        Returns a copy of the current MPS
        """
        mps_copy = MPS(self.d,self.D,self.N)
        for site in range(self.N):
            mps_copy[site] = copy.deepcopy(self[site])
        return mps_copy

    def __getitem__(self,i):
        """
        Returns the MPS tensor at site i
        """
        if not hasattr(i,'__len__'):
            return self.tensors[i]
        else:
            res = []
            for ind in range(len(i)):
                res.append(self.tensors[i[ind]])
            return res

    def __setitem__(self, site, ten):
        # Update tensor
        self.tensors[site] = ten
        # Update tensor shape
        (DL,d,DR) = self[site].shape
        self.ten_dims[site] = [DL,d,DR]
        self.d[site] = d

    def __mult__(self, const):
        """
        Multiply an MPS by a constant
        """
        mps_tmp = self.copy()
        for site in range(self.N):
            mps_tmp[site] = mps_tmp[site] * (abs(const)**(1./mps_temp.N))
            if (const < 0.) and (site == 0):
                mps_tmp[site] *= -1.
        return mps_tmp

    def norm(self):
        """
        Compute the norm of the MPS
        """
        # Create "environment" to hold contraction 
        norm_env = ones((1,1),dtype=self.dtype)
        # Move to the left, contracting env with bra and ket tensors
        for site in range(self.N):
            tmp1 = einsum('Aa,apb->Apb',copy.deepcopy(norm_env),self[site])
            norm_env = einsum('Apb,ApB->Bb',tmp1,conj(self.tensors[site]))
        # Extract and return result
        norm = norm_env[0,0]
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
        res = mps_apply_svd(self.tensors,chi)
        self.input_mps_list(res)

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
        return contract_mps(self.tensors,mps2=mps2.tensors)

    def __len__(self):
        return len(self.tensors)
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
