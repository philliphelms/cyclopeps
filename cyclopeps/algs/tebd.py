from cyclopeps.tools.params import *
from cyclopeps.tools.utils import *
from cyclopeps.tools.mps_tools import *
from cyclopeps.tools.peps_tools import *
from cyclopeps.tools.ops_tools import *
from cyclopeps.algs.simple_update import run_tebd as su
from numpy import float_,isfinite
import numpy as np

def cost_func(N,phys_b,phys_t,phys_b_new,phys_t_new,U):
    """
    Calculate the cost function to check similarity between new and
    old peps tensors

    Args:
        N : Rank-4 Array
            The Rank 4 environment tensor
        phys_b : Rank-3 Array
            Rank 3 tensor for the bottom site in the peps, from before
            the time evolution
        phys_t : Rank-3 Array
            Rank 3 tensor for the top site in the peps, from before
            the time evolution.
        phys_b_new : Rank-3 Array
            Rank 3 tensor for the bottom site in the peps (which is
            currently being optimized in als procedure)
        phys_t_new : Rank-3 Array
            Rank 3 tensor for the top site in the peps (which is
            currently being optimized in als procedure)
        U : Rank-3 Array
            The time evolution gate between the two sites

    Returns :
        d : float
            The cost function's value

    Note:
        implemented according to Equation 14 in
        https://arxiv.org/pdf/1503.05345.pdf
    """

    # Calculate a^+ * R * a
    aRa = calc_local_op(phys_b_new,phys_t_new,N,None,normalize=False)
    # Calculate a^+ * S
    # PH - Need to make sure the tensors aren't flipped for time evolution
    Sa = calc_local_op(phys_b_new,phys_t_new,N,U,phys_b_ket=phys_b,phys_t_ket=phys_t,normalize=False)
    # Calculate S^+ * a
    # (currently just using a copy)
    aS  = Sa

    return aRa-aS-Sa

def optimize_bottom(N,phys_b,phys_t,phys_b_new,phys_t_new,eH):
    """
    Note:
        implemented according to Section II.B of
        https://arxiv.org/pdf/1503.05345.pdf
    """

    # Calculate R
    tmp = einsum('DPU,dPu->DUdu',phys_t_new,phys_t_new.copy().conj())
    R   = einsum('DUdu,AaUu->ADad',tmp,N)

    # Calculate S
    tmp = einsum('APD,AaBb->DPaBb',phys_b,N)
    tmp = einsum('DPaBb,DQB->PaQb',tmp,phys_t)
    if len(tmp.legs[0]) == 2:
        # Then thermal state
        tmp.unmerge_ind(2)
        tmp.unmerge_ind(0)
        tmp = einsum('PxaQyb,PQpq->abpxqy',tmp,eH)
        tmp.merge_inds([4,5])
        tmp.merge_inds([2,3])
    else:
        # Then regular state
        tmp = einsum('PaQb,PQpq->abpq',tmp,eH)
    S   = einsum('abpq,dqb->apd',tmp,phys_t_new)

    # Take inverse of R
    R_ = R.square_inv()

    # Compute phys_b_new
    phys_b_new = einsum('ADad,apd->ApD',R_,S)

    # Return Results
    return phys_b_new

def svd_evolve(phys_b,phys_t,eH):
    """
    Do time evolution by applying gate, then
    doing svd and truncation to separate sites again
    """
    # Do time evolution
    tmp = einsum('aPb,bQc->aPQc',phys_b,phys_t)
    if len(tmp.legs[1]) == 2:
        # Thermal state
        tmp.unmerge_ind(2)
        tmp.unmerge_ind(1)
        result = einsum('aPxQyc,PQpq->apxqyc',tmp,eH)
        result.merge_inds([1,2])
        result.merge_inds([2,3])
    else:
        # Regular peps state
        result = einsum('aPQc,PQpq->apqc',tmp,eH)
    
    # Do svd & truncation
    if phys_b.sym is None:
        D = phys_b.ten.shape[phys_b.legs[2][0]]
    else:
        D = len(phys_b.ten.sym[1][phys_b.legs[2][0]])*phys_b.ten.shape[phys_b.legs[2][0]]
    U,S,V = result.svd(2,
                       truncate_mbd=D,
                       return_ent=False,
                       return_wgt=False)
    # Absorb singular values
    U = einsum('ijk,kl->ijl',U,S.sqrt())
    V = einsum('ij,jkl->ikl',S.sqrt(),V)

    return U,V

def optimize_top(N,phys_b,phys_t,phys_b_new,phys_t_new,eH,use_inv=False):
    """
    Note:
        implemented according to Section II.B of
        https://arxiv.org/pdf/1503.05345.pdf
    """

    # Calculate R
    #tmp = einsum('DPU,dPu->DUdu',phys_b_new,conj(copy.deepcopy(phys_b_new)))
    tmp = einsum('DPU,dPu->DUdu',phys_b_new,phys_b_new.copy().conj())
    R = einsum('DUdu,DdAa->UAua',tmp,N)

    # Calculate S
    tmp = einsum('UQA,DdAa->UQDda',phys_t,N)
    tmp = einsum('UQDda,DPU->PQad',tmp,phys_b)
    if len(tmp.legs[0]) == 2:
        tmp.unmerge_ind(1)
        tmp.unmerge_ind(0)
        tmp = einsum('PxQyad,PQpq->pxqyad',tmp,eH)
        tmp.merge_inds([2,3])
        tmp.merge_inds([0,1])
    else:
        tmp = einsum('PQad,PQpq->pqad',tmp,eH)
    S   = einsum('pqad,dpu->uqa',tmp,phys_b_new)

    # Solve least squares problem
    # Take inverse of R
    R_ = R.square_inv()

    # Compute new phys_t_new
    phys_t_new = einsum('uaUA,uqa->UqA',R_,S)

    # Return Results
    return phys_t_new

def split_sites(comb,mbd):
    """
    Given two combined (reduced) sites,
    split them back into two separated
    tensor sites via svd
    """
    # Do the SVD Decomposition
    site1,sings,site2 = comb.svd(2,
                                 truncate_mbd=mbd,
                                 return_ent=False,
                                 return_wgt=False)

    # Do some renormalization
    sings /= einsum('ij,jk->ik',sings,sings).sqrt().to_val()

    # Absorb singular values into sites
    site1 = einsum('ijk,kl->ijl',site1,sings.sqrt())
    site2 = einsum('ij,jkl->ikl',sings.sqrt(),site2)

    # Return the result
    return site1,site2

def simple_update_init_guess(phys_b,phys_t,eH,mbd):
    """
    Create an initial guess for the ALS procedure
    via a simple update style update
    """
    # Apply time evolution gate
    tmp = einsum('DPA,AQU->DPQU',phys_b,phys_t)
    if len(tmp.legs[1]) == 2:
        # Thermal State time evolution
        tmp.unmerge_ind(2)
        tmp.unmerge_ind(1)
        tmp = einsum('DPxQyU,PQpq->DpxqyU',tmp,eH)
        tmp.merge_inds([1,2])
        tmp.merge_inds([2,3])
    else:
        # Regular state time evolution
        tmp = einsum('DPQU,PQpq->DpqU',tmp,eH)

    # Split via SVD
    phys_b,phys_t = split_sites(tmp,mbd)

    # Return Results
    return phys_b,phys_t

def noiseless_als(phys_b,phys_t,N,eH,mbd,als_iter=100,als_tol=1e-10,stablesplit=True):
    """
    Do the Alternating least squares procedure, using the current
    physical index-holding tensors as the initial guess
    """
    # Create initial guesses for resulting tensors
    #tmpprint('\t\t\t\tSU Initial Guess')
    phys_b_new,phys_t_new = simple_update_init_guess(phys_b,phys_t,eH,mbd)

    # Initialize cost function
    cost_prev = cost_func(N,phys_b,phys_t,phys_b_new,phys_t_new,eH)

    for i in range(als_iter):

        # Optimize Bottom Site
        #tmpprint('\t\t\t\tOptimize Bottom')
        phys_b_new = optimize_bottom(N,phys_b,phys_t,phys_b_new,phys_t_new,eH)

        # Optimize Top Site
        #tmpprint('\t\t\t\tOptimize Top')
        phys_t_new = optimize_top(N,phys_b,phys_t,phys_b_new,phys_t_new,eH)

        # Split singular values to stabilize
        #tmpprint('\t\t\t\tStable Split')
        if stablesplit:
            comb = einsum('DPA,AQU->DPQU',phys_b_new,phys_t_new)
            phys_b_new,phys_t_new = split_sites(comb,mbd)

        # Check for convergence
        #tmpprint('\t\t\t\tCost Function')
        cost = cost_func(N,phys_b,phys_t,phys_b_new,phys_t_new,eH)
        if (abs(cost) < als_tol) or (abs((cost-cost_prev)/cost) < als_tol):
            break
        else:
            cost_prev = cost

    # Return result
    return phys_b_new,phys_t_new

def noisy_als(phys_b,phys_t,N,eH,als_iter=100,als_tol=1e-10,stablesplit=True):
    """
    Do the Alternating least squares procedure, using the current
    physical index-holding tensors (with some noise) as the initial guess
    """
    # Create initial guesses for resulting tensors
    phys_b_new,phys_t_new = simple_update_init_guess(phys_b,phys_t,eH,mbd)

    # Add a bit of noise
    noise_b = phys_b_new.copy()
    noise_t = phys_t_new.copy()
    noise_b.randomize()
    noise_t.randomize()
    phys_b_new += 1e-5*noise_b
    phys_t_new += 1e-5*noise_t

    # Initialize cost function
    cost_prev = cost_func(N,phys_b,phys_t,phys_b_new,phys_t_new,eH)

    for i in range(als_iter):

        # Optimize Bottom Site
        phys_b_new = optimize_bottom(N,phys_b,phys_t,phys_b_new,phys_t_new,eH)

        # Optimize Top Site
        phys_t_new = optimize_top(N,phys_b,phys_t,phys_b_new,phys_t_new,eH)

        # Split singular values to stabilize
        if stablesplit:
            comb = einsum('DPA,AQU->DPQU',phys_b_new,phys_t_new)
            phys_b_new,phys_t_new = split_sites(comb,mbd)

        # Check for convergence
        cost = cost_func(N,phys_b,phys_t,phys_b_new,phys_t_new,eH)
        if (abs(cost) < als_tol) or (abs((cost-cost_prev)/cost) < als_tol):
            break
        else:
            cost_prev = cost

    # Return result
    return phys_b_new,phys_t_new

def alternating_least_squares(phys_b,phys_t,N,eH,mbd,als_iter=100,als_tol=1e-10):
    """
    Do alternating least squares to determine best tensors
    to represent time evolved tensors at smaller bond dimensions
    """
    try:
        return noiseless_als(phys_b,phys_t,N,eH,mbd,als_iter=als_iter,als_tol=als_tol)
    except:
        # If als fails, then there are likely many zeros, so we expect
        # the time evolved tensors to be low rank, meaning doing a simple
        # update style evolution will provide a better initial guess
        phys_b,phys_t = svd_evolve(phys_b,phys_t,eH)
        return noiseless_als(phys_b,phys_t,N,eH,mbd,als_iter=als_iter,als_tol=als_tol)

def make_equal_distance(peps1,peps2,mbd):
    """
    Multiplying nearest neighbor peps tensors together and resplitting them
    so that the singular values are equally split between the two tensors
    """
    # Pull the physical index off each tensor
    peps1 = peps1.transpose([0,1,3,2,4])
    (ub,sb,vb) = peps1.svd(3,return_ent=False,return_wgt=False)
    phys_b = einsum('aA,APU->aPU',sb,vb)
    peps2 = peps2.transpose([1,2,0,3,4])
    (ut,st,vt) = peps2.svd(2,return_ent=False,return_wgt=False)
    phys_t = einsum('DPa,aA->DPA',ut,st)
    vt = vt.transpose([1,0,2,3])

    # Combine the two reduced tensors
    theta = einsum('aPU,UQb->aPQb',phys_b,phys_t)

    # Take svd of result and truncate D if needed
    (u,s,v) = theta.svd(2,truncate_mbd=mbd,return_ent=False,return_wgt=False)
    phys_b = einsum('aPU,Uu->aPu',u,s.sqrt())
    phys_t = einsum('dD,DPa->dPa',s.sqrt(),v)

    # Recombine the tensors:
    peps1 = einsum('LDRa,aPU->LDPRU',ub,phys_b)
    peps2 = einsum('DPa,LaRU->LDPRU',phys_t,vt)

    # Try to shrink norm by multiplying peps1 and peps2 by constants
    peps1 /= peps1.abs().max()
    peps2 /= peps2.abs().max()

    # Return results
    return peps1,peps2

def tebd_step_single_col(peps_col,step_size,left_bmpo,right_bmpo,ham,mbd,als_iter=100,als_tol=1e-10):
    """
    """
    # Calculate top and bottom environments
    #tmpprint('\t\tCalculating top envs')
    top_envs = calc_top_envs(peps_col,left_bmpo,right_bmpo)
    #tmpprint('\t\tCalculating bot envs')
    bot_envs = calc_bot_envs(peps_col,left_bmpo,right_bmpo)

    # Loop through rows in the column
    E = peps_col[0].backend.zeros(len(ham),dtype=peps_col[0].dtype)
    for row in range(len(ham)):
        #tmpprint('\t\tDoing TEBD on sites ({},{})'.format(row,row+1))

        # Calculate environment aroudn reduced tensors
        #tmpprint('\t\t\tCalculating Environment')
        peps_b,phys_b,phys_t,peps_t,_,_,_,_,N = calc_N(row,peps_col,left_bmpo,right_bmpo,top_envs,bot_envs)

        # Take the exponential of the hamiltonian
        #tmpprint('\t\t\tExponentiating Hamiltonian')
        eH = exp_gate(ham[row],-step_size)

        # Do alternating least squares to find new peps tensors
        #tmpprint('\t\t\tDoing ALS')
        phys_b,phys_t = alternating_least_squares(phys_b,phys_t,N,eH,mbd,als_iter=als_iter,als_tol=als_tol)

        # Calculate Energy & Norm
        #tmpprint('\t\t\tCalculating Local Energy')
        E[row],norm = calc_local_op(phys_b,phys_t,N,ham[row],return_norm=True)

        # Update peps_col tensors
        #tmpprint('\t\t\tUpdating peps tensors')
        peps_col[row]   = einsum('LDRa,aPU->LDPRU',peps_b,phys_b)
        peps_col[row+1] = einsum('DPa,LaRU->LDPRU',phys_t,peps_t)

        # Combine and equally split the two tensors
        #tmpprint('\t\t\tMaking peps tensors stable')
        peps_col[row],peps_col[row+1] = make_equal_distance(peps_col[row],peps_col[row+1],mbd)
        #tmpprint('peps_col[i]: {}'.format(peps_col[row].ten.array.shape))
        #tmpprint('peps_col[i+1]: {}'.format(peps_col[row].ten.array.shape))

        # Update top and bottom environments
        #tmpprint('\t\t\tUpdating Bottom environment')
        if row == 0: prev_env = None
        else: prev_env = bot_envs[row-1]
        bot_envs[row] = update_bot_env(peps_col[row],
                                       peps_col[row].conj(),
                                       left_bmpo[2*row],
                                       left_bmpo[2*row+1],
                                       right_bmpo[2*row],
                                       right_bmpo[2*row+1],
                                       prev_env)

        # Normalize everything (to try to avoid some errors)
        #tmpprint('\t\t\tTrying to normalize')
        norm_fact = bot_envs[row].abs().max()
        bot_envs[row] /= norm_fact
        peps_col[row] /= norm_fact**(1./2.)
        peps_col[row+1] /= norm_fact**(1./2.)

    # Return the result
    return E,peps_col

def tebd_step_col(peps,ham,step_size,mbd,chi=None,als_iter=100,als_tol=1e-10):
    """
    """
    # Figure out peps size
    (Nx,Ny) = peps.shape

    # Compute the boundary MPOs
    #tmpprint('\tCalculating Boundary MPOs')
    right_bmpo = calc_right_bound_mpo(peps, 0,chi=chi,return_all=True)
    left_bmpo  = [None]*(Nx-1)
    ident_bmpo = identity_mps(len(right_bmpo[0]),
                              dtype=peps[0][0].dtype,
                              sym=(peps[0][0].sym is not None),
                              backend=peps.backend)

    # Loop through all columns
    E = peps.backend.zeros((len(ham),len(ham[0])),dtype=peps[0][0].dtype)
    for col in range(Nx):
        #tmpprint('\tDoing TEBD In column {}'.format(col))
        # Take TEBD Step
        if col == 0:
            res = tebd_step_single_col(peps[col],
                                       step_size,
                                       ident_bmpo,
                                       right_bmpo[col],
                                       ham[col],
                                       mbd,
                                       als_iter=als_iter,
                                       als_tol=als_tol)
        elif col == Nx-1:
            res = tebd_step_single_col(peps[col],
                                       step_size,
                                       left_bmpo[col-1],
                                       ident_bmpo,
                                       ham[col],
                                       mbd,
                                       als_iter=als_iter,
                                       als_tol=als_tol)
        else:
            res = tebd_step_single_col(peps[col],
                                       step_size,
                                       left_bmpo[col-1],
                                       right_bmpo[col],
                                       ham[col],
                                       mbd,
                                       als_iter=als_iter,
                                       als_tol=als_tol)
        E[col,:] = res[0]
        peps[col] = res[1]

        # Update left boundary tensors
        if col == 0:
            left_bmpo[col] = update_left_bound_mpo(peps[col], None, chi=chi)
        elif col != Nx-1:
            left_bmpo[col] = update_left_bound_mpo(peps[col], left_bmpo[col-1], chi=chi)

    # Return result
    return E,peps

def tebd_step(peps,ham,step_size,mbd,chi=None,als_iter=100,als_tol=1e-10,print_prepend=''):
    """
    """
    # Columns ----------------------------------
    #tmpprint('Doing Column Interactions')
    Ecol,peps = tebd_step_col(peps,ham[0],step_size,mbd,chi=chi,als_iter=als_iter,als_tol=als_tol)
    # Rows -------------------------------------
    peps.rotate(clockwise=True)
    #tmpprint('Doing Row Interactions')
    Erow,peps = tebd_step_col(peps,ham[1],step_size,mbd,chi=chi,als_iter=als_iter,als_tol=als_tol)
    peps.rotate(clockwise=False)
    # Return results ---------------------------
    mpiprint(5,print_prepend+'Column energies =\n{}'.format(Ecol))
    mpiprint(5,print_prepend+'Row energies =\n{}'.format(Erow))
    E = peps.backend.sum(Ecol)+peps.backend.sum(Erow)
    return E,peps

def tebd_steps(peps,ham,step_size,n_step,conv_tol,mbd,chi=None,chi_norm=10,chi_op=None,als_iter=100,als_tol=1e-10,print_prepend='',save_all_steps=False):
    """
    """
    nSite = len(peps)*len(peps[0])

    # Compute Initial Energy
    mpiprint(3,print_prepend+'Calculation Initial Energy/site')
    Eprev = peps.calc_op(ham,chi=chi_op)
    mpiprint(0,print_prepend+'Initial Energy/site = {}'.format(Eprev/nSite))

    # Do a single tebd step
    for iter_cnt in range(n_step):

        # Do TEBD Step
        _,peps = tebd_step(peps,ham,step_size,mbd,chi=chi,als_iter=als_iter,als_tol=als_tol,print_prepend=print_prepend)

        # Normalize just in case
        peps.normalize(chi=chi_norm)

        # Save PEPS
        if save_all_steps: 
            peps.save(fname=peps.fname+'_iter{}'.format(iter_cnt))
        else:
            peps.save()
        
        # Compute Resulting Energy
        E = peps.calc_op(ham,chi=chi_op)

        # Check for convergence
        mpiprint(0,print_prepend+'Energy/site = {}'.format(E/nSite))
        if abs((E-Eprev)/E) < conv_tol:
            mpiprint(3,'Converged E = {} to an accuracy of ~{}'.format(E,abs(E-Eprev)))
            converged = True
            break
        else:
            Eprev = E
            converged = False
    return E,peps

def run_tebd(Nx,Ny,d,ham,
             peps=None,
             backend='numpy',
             D=3,
             Zn=None,
             chi=10,
             su_chi=10,
             chi_norm=None,
             chi_op=None,
             thermal=False,
             exact_norm_tol=20,
             norm_tol=0.1,
             singleLayer=True,
             max_norm_iter=20,
             dtype=float_,
             step_size=[0.1,0.01,0.001],
             su_step_size=None,
             n_step=5,
             su_n_step=None,
             conv_tol=1e-8,
             su_conv_tol=1e-4,
             als_iter=5,
             als_tol=1e-10,
             peps_fname=None,
             peps_fdir='./',
             save_all_steps=False,
             print_prepend=''):
    """
    Run the TEBD algorithm for a PEPS

    Args:
        Nx : int
            Lattice size in x-direction
        Ny : int
            Lattice size in y-direction
        d : int
            Local physical bond dimension
        ham : 3D array
            The suzuki-trotter decomposition of the
            Hamiltonian. An example of how this is constructed
            for the ising transverse field model
            is found in /mpo/itf.py

    Kwargs:
        peps : PEPS object
            The initial guess for the PEPS. If this is not
            provided, then a random peps will be initialized,
            then a few iterations will be performed using
            the simple update algorithm to arrive at a good
            initial guess.
            Note that the bond dimension D should be the same
            as the initial calculation bond dimension, since
            no bond reduction or initial increase of bond dimension
            is currently implemented.
        backend : str
            The tensor backend to be used. 
            Current options are 'numpy' or 'ctf'
        D : int
            The maximum bond dimension (may be a list of
            maximum bond dimensions, and a loop will be
            performed where it is slowly incremented)
        Zn : int
            The Zn symmetry of the PEPS. 
            If None, then a dense, non-symmetric PEPS will be used.
        chi : int
            The boundary mpo maximum bond dimension
        su_chi : int
            The boundary mpo maximum bond dimension for computing
            the energy in the simple update initial guess generation
        chi_norm : int
            The boundary mpo maximum bond dimension to be used
            when normalizing the peps
        chi_op : int
            The boundary mpo maximum bond dimension to be used
            when calculating operator expectation values (default 
            uses chi)
        thermal : bool
            Whether to do the fu algorithm with a thermal state, i.e.
            two physical indices
        norm_tol : float
            How close to 1. the norm should be before
            exact arithmetic is used in the normalization
            procedure. See documentation of
            peps_tool.normalize_peps() function for more details.
        singleLayer : bool
            Whether to use a single layer environment
            (currently only option implemented)
        max_norm_iter : int
            The maximum number of normalization iterations
        dtype : dtype
            The data type for the PEPS
        step_size : float
            The trotter step size, may be a list of
            step sizes
        su_step_size : float
            The trotter step size for the simple update procedure 
            may be a list of step sizes
        n_step : int
            The number of steps to be taken for each
            trotter step size. If it is a list, then
            len(step_size) == len(n_step) and
            len(D) == len(n_step) must both be True.
        su_n_step : int
            The number of steps to be taken for each
            trotter step size in the simple update procedure.
        conv_tol : float
            The convergence tolerance
        su_conv_tol : float
            The convergence tolerance for the simple update procedure.
        peps_fname : str
            The name of the saved peps file
        peps_fdir : str
            The location where the peps will be saved
        print_prepend : str
            What to add to the beginning of all printed results, 
            default is nothing
    """
    t0 = time.time()
    mpiprint(0,'\n\n'+print_prepend+'Starting TEBD Calculation')
    mpiprint(0,print_prepend+'#'*50)

    # Ensure the optimization parameters, namely the
    # bond dimension, trotter step size, and number
    # of trotter steps are compatable.
    if hasattr(D,'__len__'):
        n_calcs = len(D)
    elif hasattr(step_size,'__len__'):
        n_calcs = len(step_size)
    elif hasattr(n_step,'__len__'):
        n_calcs = len(n_step)
    elif hasattr(conv_tol,'__len__'):
        n_calcs = len(conv_tol)
    elif hasattr(chi,'__len__'):
        n_calcs = len(chi)
    else:
        D = [D]
        step_size = [step_size]
        n_step = [n_step]
        conv_tol = [conv_tol]
        chi = [chi]
    if not hasattr(D,'__len__'):
        D = [D]*n_calcs
    if not hasattr(step_size,'__len__'):
        step_size = [step_size]*n_calcs
    if not hasattr(n_step,'__len__'):
        n_step = [n_step]*n_calcs
    if not hasattr(conv_tol,'__len__'):
        conv_tol = [conv_tol]*n_calcs
    if not hasattr(chi,'__len__'):
        chi = [chi]*n_calcs

    # Create a random peps (if one is not provided)
    if peps is None:
        if su_step_size is None: su_step_size = step_size
        if su_n_step is None: su_n_step = n_step
        if thermal:
            peps = PEPS(Nx=Nx,
                        Ny=Ny,
                        d=d,
                        D=D[0],
                        chi=chi[0],
                        chi_norm=chi_norm,
                        Zn=Zn,
                        thermal=thermal,
                        backend=backend,
                        exact_norm_tol=exact_norm_tol,
                        norm_tol=norm_tol,
                        singleLayer=singleLayer,
                        max_norm_iter=max_norm_iter,
                        dtype=dtype,
                        fname=peps_fname,
                        fdir=peps_fdir)
        else:
            _,peps = su(Nx,Ny,d,ham,
                        D=D[0],
                        Zn=Zn,
                        chi=su_chi,
                        chi_norm=chi_norm,
                        backend=backend,
                        exact_norm_tol=exact_norm_tol,
                        norm_tol=norm_tol,
                        singleLayer=singleLayer,
                        max_norm_iter=max_norm_iter,
                        dtype=dtype,
                        step_size=su_step_size,
                        n_step=su_n_step,
                        conv_tol=su_conv_tol,
                        peps_fname=peps_fname,
                        peps_fdir=peps_fdir)
            peps.absorb_lambdas()

    # Absorb lambda tensors if canonical
    if peps.ltensors is not None:
        peps.absorb_lambdas()
    
    # Make sure the peps is normalized
    peps.normalize(chi=chi_norm)

    # Loop over all (bond dims/step sizes/number of steps)
    for Dind in range(len(D)):

        mpiprint(0,'\n'+print_prepend+'FU Calculation for (D,chi,dt) = ({},{},{})'.format(D[Dind],chi[Dind],step_size[Dind]))

        # Do a tebd evolution for given step size
        E,peps = tebd_steps(peps,
                            ham,
                            step_size[Dind],
                            n_step[Dind],
                            conv_tol[Dind],
                            D[Dind],
                            chi = chi[Dind],
                            chi_norm = chi_norm,
                            als_iter=als_iter,
                            als_tol=als_tol,
                            print_prepend=print_prepend,
                            save_all_steps=save_all_steps)

    # Print out results
    mpiprint(0,'\n\n'+print_prepend+'#'*50)
    mpiprint(0,print_prepend+'FU TEBD Complete')
    mpiprint(0,print_prepend+'-------------')
    mpiprint(0,print_prepend+'Total time = {} s'.format(time.time()-t0))
    mpiprint(0,print_prepend+'Per Site Energy = {}'.format(E/(Nx*Ny)))

    return E,peps
