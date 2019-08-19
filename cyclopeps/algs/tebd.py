from cyclopeps.tools.params import *
from cyclopeps.tools.utils import *
from cyclopeps.tools.peps_tools import *
from cyclopeps.tools.ops_tools import *
from numpy import float_
import copy 

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
    # Copy tensors
    phys_b = copy.copy(phys_b)
    phys_t = copy.copy(phys_t)
    phys_b_new = copy.copy(phys_b_new)
    phys_t_new = copy.copy(phys_t_new)

    # Calculate R
    tmp = einsum('DPU,dPu->DUdu',phys_t_new,conj(copy.copy(phys_t_new)))
    R   = einsum('DUdu,AaUu->ADad',tmp,N)

    # Calculate S
    tmp = einsum('APD,AaBb->DPaBb',phys_b,N)
    tmp = einsum('DPaBb,DQB->PaQb',tmp,phys_t)
    tmp = einsum('PaQb,PQpq->abpq',tmp,eH)
    S   = einsum('abpq,dqb->apd',tmp,phys_t_new)

    # Take inverse of R
    (n1,n2,n3,n4) = R.shape
    R = reshape(R,(n1*n2,n3*n4))
    R_ = inv(R)
    R_ = reshape(R_,(n1,n2,n3,n4))

    # Compute phys_b_new
    # PH - Might be adAD instead of ADad
    phys_b_new = einsum('ADad,apd->ApD',R_,S)
    #phys_b_new = einsum('adAD,apd->ApD',R_,S)

    # Return Results
    return phys_b_new

def optimize_top(N,phys_b,phys_t,phys_b_new,phys_t_new,eH):
    """
    Note:
        implemented according to Section II.B of
        https://arxiv.org/pdf/1503.05345.pdf
    """
    # Copy tensors
    phys_b = copy.copy(phys_b)
    phys_t = copy.copy(phys_t)
    phys_b_new = copy.copy(phys_b_new)
    phys_t_new = copy.copy(phys_t_new)

    # Calculate R
    tmp = einsum('DPU,dPu->DUdu',phys_b_new,conj(copy.copy(phys_b_new)))
    R = einsum('DUdu,DdAa->UAua',tmp,N)

    # Calculate S
    tmp = einsum('UQA,DdAa->UQDda',phys_t,N)
    tmp = einsum('UQDda,DPU->PQad',tmp,phys_b)
    tmp = einsum('PQad,PQpq->pqad',tmp,eH)
    S   = einsum('pqad,dpu->uqa',tmp,phys_b_new)

    # Take inverse of R
    (n1,n2,n3,n4) = R.shape
    R = reshape(R,(n1*n2,n3*n4))
    R_ = inv(R)
    R_ = reshape(R_,(n1,n2,n3,n4))

    # Compute new phys_t_new
    # PH - Might be 
    #phys_t_new = einsum('UAua,uqa->UqA',R_,S)
    phys_t_new = einsum('uaUA,uqa->UqA',R_,S)

    # Return Results
    return phys_t_new

def alternating_least_squares(phys_b,phys_t,N,eH,als_iter=100,als_tol=1e-10):
    """
    """
    # Copy tensors (we never change phys_b or phys_t)
    phys_b_new = copy.copy(phys_b)
    phys_t_new = copy.copy(phys_t)

    # Initialize cost function
    cost_prev = cost_func(N,phys_b,phys_t,phys_b_new,phys_t_new,eH)
    #print('Cost = {}'.format(cost_prev))

    for i in range(als_iter):
        
        # Optimize Bottom Site
        phys_b_new = optimize_bottom(N,phys_b,phys_t,phys_b_new,phys_t_new,eH)

        # Optimize Top Site
        phys_t_new = optimize_top(N,phys_b,phys_t,phys_b_new,phys_t_new,eH)

        # Check for convergence
        cost = cost_func(N,phys_b,phys_t,phys_b_new,phys_t_new,eH)
        #print('Cost = {}'.format(cost))
        if (abs(cost) < als_tol) or (abs((cost-cost_prev)/cost) < als_tol):
        #if (abs(cost) > abs(cost_prev)) or (abs(cost) < als_tol) or (abs((cost-cost_prev)/cost) < als_tol):
            break
        else:
            cost_prev = cost
    
    # Return result
    return phys_b_new,phys_t_new

def exp_ham(ham,a=1.):
    """
    Take the exponential of the Hamiltonian
    """
    ham = copy.copy(ham)
    d = ham.shape[0]
    ham = reshape(ham,(d**2,d**2))
    eH = expm(ham,a)
    eH = reshape(eH,(d,d,d,d))
    return eH

def make_equal_distance(peps1,peps2,D=None):
    """
    Multiplying nearest neighbor peps tensors together and resplitting them
    so that the singular values are equally split between the two tensors
    """
    # Set maximum bond dim as current maximum bond dim
    if D is None:
        _,_,_,_,D = peps1.shape

    # Pull the physical index off each tensor
    peps1 = einsum('LDPRU->LDRPU',peps1)
    (ub,sb,vb) = mps_tools.svd_ten(peps1,3,return_ent=False,return_wgt=False)
    phys_b = einsum('a,aPU->aPU',sb,vb)
    peps2 = einsum('LDPRU->DPLRU',peps2)
    (ut,st,vt) = mps_tools.svd_ten(peps2,2,return_ent=False,return_wgt=False)
    phys_t = einsum('DPa,a->DPa',ut,st)
    vt = einsum('aLRU->LaRU',vt)

    # Combine the two reduced tensors
    theta = einsum('aPU,UQb->aPQb',phys_b,phys_t)

    # Take svd of result and truncate D if needed
    (u,s,v) = mps_tools.svd_ten(theta,2,truncate_mbd=D,return_ent=False,return_wgt=False)
    phys_b = einsum('aPU,U->aPU',u,sqrt(s))
    phys_t = einsum('D,DPa->DPa',sqrt(s),v)

    # Recombine the tensors:
    peps1 = einsum('LDRa,aPU->LDPRU',ub,phys_b)
    peps2 = einsum('DPa,LaRU->LDPRU',phys_t,vt)

    # Return results
    return peps1,peps2

def tebd_step_single_col(peps_col,step_size,left_bmpo,right_bmpo,ham,als_iter=100,als_tol=1e-10):
    """
    """
    # Calculate top and bottom environments
    top_envs = calc_top_envs(peps_col,left_bmpo,right_bmpo)
    bot_envs = calc_bot_envs(peps_col,left_bmpo,right_bmpo)

    # Loop through rows in the column
    E = zeros(len(ham),dtype=peps_col[0].dtype)
    for row in range(len(ham)):
        
        # Calculate environment aroudn reduced tensors
        peps_b,phys_b,phys_t,peps_t,N = calc_N(row,peps_col,left_bmpo,right_bmpo,top_envs,bot_envs)

        # Take the exponential of the hamiltonian
        eH = exp_ham(ham[row],-step_size)

        # Do alternating least squares to find new peps tensors
        phys_b,phys_t = alternating_least_squares(phys_b,phys_t,N,eH,als_iter=als_iter,als_tol=als_tol)

        # Calculate Energy & Norm
        E[row] = calc_local_op(phys_b,phys_t,N,ham[row])

        # Update peps_col tensors
        peps_col[row]   = einsum('LDRa,aPU->LDPRU',peps_b,phys_b)
        peps_col[row+1] = einsum('DPa,LaRU->LDPRU',phys_t,peps_t)

        # Whatever equall_dis_V does
        peps_col[row],peps_col[row+1] = make_equal_distance(peps_col[row],peps_col[row+1])

        # Whatever max_ten does (just shrink an entry so the norm isn't too large)
        # PH - Implement this later

        # Update top and bottom environments
        # PH - Very lazy and expensive way to do this...
        top_envs = calc_top_envs(peps_col,left_bmpo,right_bmpo)
        bot_envs = calc_bot_envs(peps_col,left_bmpo,right_bmpo)

    # Return the result
    return E,peps_col

def tebd_step_col(peps,ham,step_size,chi=None,als_iter=100,als_tol=1e-10):
    """
    """
    # Figure out peps size
    (Nx,Ny) = peps.shape

    # Compute the boundary MPOs
    right_bmpo = calc_right_bound_mpo(peps, 0,chi=chi,return_all=True)
    left_bmpo  = calc_left_bound_mpo (peps,Nx,chi=chi,return_all=True)
    ident_bmpo = mps_tools.identity_mps(len(right_bmpo[0]),dtype=peps[0][0].dtype)

    # Loop through all columns
    E = zeros((len(ham),len(ham[0])),dtype=peps[0][0].dtype)
    for col in range(Nx):
        # Take TEBD Step
        if col == 0:
            E[col,:],peps[col] = tebd_step_single_col(peps[col],step_size,ident_bmpo,right_bmpo[col],ham[col],als_iter=als_iter,als_tol=als_tol)
        elif col == Nx-1:
            E[col,:],peps[col] = tebd_step_single_col(peps[col],step_size,left_bmpo[col-1],ident_bmpo,ham[col],als_iter=als_iter,als_tol=als_tol)
        else:
            E[col,:],peps[col] = tebd_step_single_col(peps[col],step_size,left_bmpo[col-1],right_bmpo[col],ham[col],als_iter=als_iter,als_tol=als_tol)

        # Update boundary tensors
        # PH - Very lazy and expensive way to do this...
        right_bmpo = calc_right_bound_mpo(peps, 0,chi=chi,return_all=True)
        left_bmpo  = calc_left_bound_mpo (peps,Nx,chi=chi,return_all=True)

    # Return result
    return E,peps

def tebd_step(peps,ham,step_size,chi=None,als_iter=100,als_tol=1e-10):
    """
    """
    # Columns ----------------------------------
    Ecol,peps = tebd_step_col(peps,ham[0],step_size,chi=chi,als_iter=als_iter,als_tol=als_tol)
    # Rows -------------------------------------
    peps.rotate(clockwise=True)
    Erow,peps = tebd_step_col(peps,ham[1],step_size,chi=chi,als_iter=als_iter,als_tol=als_tol)
    peps.rotate(clockwise=False)
    # Return results ---------------------------
    E = summ(Ecol)+summ(Erow)
    return E,peps

def tebd_steps(peps,ham,step_size,n_step,conv_tol,chi=None,als_iter=100,als_tol=1e-10):
    """
    """
    # Compute Initial Energy
    Eprev = peps.calc_op(ham,chi=chi)
    #print('Initial Energy = {}'.format(Eprev))

    # Do a single tebd step
    for iter_cnt in range(n_step):

        # Do TEBD Step
        E,peps = tebd_step(peps,ham,step_size,chi=chi,als_iter=als_iter,als_tol=als_tol)
        print(E)

        # Normalize just in case
        peps.normalize()
        
        # Compute Resulting Energy
        E2 = peps.calc_op(ham,chi=chi)
        
        # Check for convergence
        mpiprint(1,'Energy = {} ({})'.format(E,E2))
        if abs((E-Eprev)/E) < conv_tol:
            mpiprint(3,'Converged E = {} to an accuracy of ~{}'.format(E,abs(E-Eprev)))
            converged = True
            break
        else:
            Eprev = E
            converged = False
    return E,peps

def run_tebd(Nx,Ny,d,ham,
             D=3,chi=10,
             norm_tol=1e-5,singleLayer=True,
             max_norm_iter=100,norm_change_int=3e-2,
             dtype=float_,
             step_size=0.1,n_step=10,conv_tol=1e-8,
             als_iter=5,als_tol=1e-10):
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
        D : int
            The maximum bond dimension (may be a list of
            maximum bond dimensions, and a loop will be
            performed where it is slowly incremented)
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
        step_size : float
            The trotter step size, may be a list of 
            step sizes
        n_step : int
            The number of steps to be taken for each
            trotter step size. If it is a list, then 
            len(step_size) == len(n_step) and 
            len(D) == len(n_step) must both be True.
        conv_tol : float
            The convergence tolerance

    """
    t0 = time.time()
    mpiprint(0,'\n\nStarting TEBD Calculation')
    mpiprint(0,'#'*50)

    # Ensure the optimization parameters, namely the
    # bond dimension, trotter step size, and number
    # of trotter steps are compatable.
    # PH - Should add conv_tol and chi to this so it can be a list
    if not hasattr(D,'__len__'): 
        if (not hasattr(step_size,'__len__')) and (not hasattr(n_step,'__len__')):
            D = [D]
        elif hasattr(step_size,'__len__'):
            D = [D]*len(step_size)
            n_step = [n_step]*len(step_size)
        elif hasattr(n_step,'__len__'):
            D = [D]*len(n_step)
            step_size = [step_size]*len(n_step)
        else:
            D = [D]
            n_step = [n_step]
            step_size = [step_size]
    else:
        if hasattr(step_size,'__len__'):
            assert(len(step_size) == len(D))
        else:
            step_size = [step_size]*len(D)
        if hasattr(n_step,'__len__'):
            assert(len(n_step) == len(D))
        else:
            n_step = [n_step]*len(D)
    chi = [chi]*len(D)
    conv_tol = [conv_tol]*len(D)
    step_size = [step_size]*len(D)
    n_step = [n_step]*len(D)
    
    # Create a random peps
    peps = PEPS(Nx=Nx,Ny=Ny,d=d,D=D[0],
                chi=chi[0],norm_tol=norm_tol,
                singleLayer=singleLayer,max_norm_iter=max_norm_iter,
                norm_change_int=norm_change_int,dtype=dtype)
    
    # Loop over all (bond dims/step sizes/number of steps)
    for Dind in range(len(D)):
        
        # Do a tebd evolution for given step size
        E,peps = tebd_steps(peps,
                            ham,
                            step_size[Dind],
                            n_step[Dind],
                            conv_tol[Dind],
                            chi = chi[Dind],
                            als_iter=als_iter,
                            als_tol=als_tol)

        # Increase MBD if needed
        if len(D)-1 > Dind:
            peps.increase_mbd(D[Dind+1],chi=chi[Dind+1])

    return E

if __name__ == "__main__":
    # PEPS parameters
    Nx = 3
    Ny = 5
    d = 2
    D = 3
    chi = 10
    norm_tol = 1e-5
    # Get Hamiltonian
    from cyclopeps.ops.itf import return_op
    ham = return_op(Nx,Ny,(1.,2.))
    # Run TEBD
    E = run_tebd(Nx,Ny,d,ham,
                 D=D,chi=chi,
                 norm_tol=norm_tol,singleLayer=True,
                 max_norm_iter=100,norm_change_int=3e-2,
                 dtype=float_,
                 step_size=0.1,n_step=10)
