from cyclopeps.tools.utils import *
from cyclopeps.tools.peps_tools import *
from cyclopeps.tools.ops_tools import *
from numpy import float_
import copy 

def absorb_lambdas(row,peps_col,vert_lambdas,left_lambdas,right_lambdas):
    """
    """
    peps1 = peps_col[row].copy()
    peps2 = peps_col[row+1].copy()
    # Absorb Bottom lambda
    if not (row == 0):
        peps1 = einsum('ldpru,dD->lDpru',peps1,vert_lambdas[row-1])
    # Absorb left lambdas
    if left_lambdas is not None:
        peps1 = einsum('ldpru,lL->Ldpru',peps1,left_lambdas[row])
        peps2 = einsum('ldpru,lL->Ldpru',peps2,left_lambdas[row+1])
    # Absorb right lambdas
    if right_lambdas is not None:
        peps1 = einsum('ldpru,rR->ldpRu',peps1,right_lambdas[row])
        peps2 = einsum('ldpru,rR->ldpRu',peps2,right_lambdas[row+1])
    # Absorb Top lambda
    if not (row == len(peps_col)-2):
        peps2 = einsum('ldpru,uU->ldprU',peps2,vert_lambdas[row+1])
    # Absorb middle lambda
    peps1 = einsum('ldpru,uU->ldprU',peps1,vert_lambdas[row])
    return peps1,peps2

def separate_sites(combined_sites,D):
    """
    """
    # Do the SVD Decomposition
    peps1,Lambda,peps2 = combined_sites.svd(4,
                                            truncate_mbd=D,
                                            return_ent=False,
                                            return_wgt=False)
    # Do some renormalization (just to keep numbers reasonable)
    Lambda /= einsum('ij,jk->ik',Lambda,Lambda).sqrt().to_val()
    # Reshape the results
    peps2 = peps2.transpose([1,0,2,3,4])
    return peps1,Lambda,peps2

def remove_lambdas(row,peps_col,vert_lambdas,left_lambdas,right_lambdas):
    """
    """
    peps1 = peps_col[row].copy()
    peps2 = peps_col[row+1].copy()
    # Absorb Bottom lambda
    if not (row == 0):
        peps1 = einsum('ldpru,dD->lDpru',peps1,1./vert_lambdas[row-1])
    # Absorb left lambdas
    if left_lambdas is not None:
        peps1 = einsum('ldpru,lL->Ldpru',peps1,1./left_lambdas[row])
        peps2 = einsum('ldpru,lL->Ldpru',peps2,1./left_lambdas[row+1])
    # Absorb right lambdas
    if right_lambdas is not None:
        peps1 = einsum('ldpru,rR->ldpRu',peps1,1./right_lambdas[row])
        peps2 = einsum('ldpru,rR->ldpRu',peps2,1./right_lambdas[row+1])
    # Absorb Top lambda
    if not (row == len(peps_col)-2):
        peps2 = einsum('ldpru,uU->ldprU',peps2,1./vert_lambdas[row+1])

    # Put them back in the list
    peps_col[row] = peps1
    peps_col[row+1] = peps2

    # Return result
    return peps_col

def tebd_step_single_col(peps_col,vert_lambdas,left_lambdas,right_lambdas,step_size,ham):
    """
    """

    # Loop through rows in the column
    E = zeros(len(ham),dtype=peps_col[0].dtype)
    for row in range(len(ham)):

        # Absorb Lambdas into Gamma tensors
        peps1,peps2 = absorb_lambdas(row,peps_col,vert_lambdas,left_lambdas,right_lambdas)

        # Take the exponential of the hamiltonian
        eH = exp_gate(ham[row],-step_size)

        # Apply Time Evolution
        if DEBUG:
            comparison = einsum('ldpru,LuPRU->ldprLPRU',peps1,peps2)
        tmp = einsum('ldpru,LuPRU->ldprLPRU',peps1,peps2)
        result = einsum('ldprLPRU,pPqQ->ldqrLQRU',tmp,eH)
        #result = einsum('ldpru,LuPRU,pPqQ->ldqrLQRU',peps1,peps2,eH)
        if DEBUG:
            print(summ(abss(comparison-result)))

        # Perform SVD
        D = peps1.shape[4]
        peps1,Lambda,peps2 = separate_sites(result,D)
        if DEBUG:
            comparison2 = einsum('ldpru,LuPRU,u->ldprLPRU',peps1,peps2,Lambda)
            print(summ(abss(comparison2-result)))
        
        # Put result back into vectors
        vert_lambdas[row] = Lambda
        peps_col[row]   = peps1
        peps_col[row+1] = peps2

        # Remove Lambdas
        peps_col = remove_lambdas(row,peps_col,vert_lambdas,left_lambdas,right_lambdas)


        peps1,peps2 = absorb_lambdas(row,peps_col,vert_lambdas,left_lambdas,right_lambdas)
        if DEBUG:
            comparison = einsum('ldpru,LuPRU->ldprLPRU',peps1,peps2)
            print(summ(abss(comparison-result)))

    # Return the result
    return peps_col,vert_lambdas

def tebd_step_col(peps,ham,step_size):
    """
    """
    # Figure out peps size
    (Nx,Ny) = peps.shape

    # Loop through all columns
    E = zeros((len(ham),len(ham[0])),dtype=peps[0][0].dtype)
    for col in range(Nx):
        # Take TEBD Step
        if col == 0:
            peps[col],peps.ltensors[0][col] = tebd_step_single_col(peps[col],
                                                                   peps.ltensors[0][col],
                                                                   None,
                                                                   peps.ltensors[1][col],
                                                                   step_size,
                                                                   ham[col])
        elif col == (Nx-1):
            peps[col],peps.ltensors[0][col] = tebd_step_single_col(peps[col],
                                                                   peps.ltensors[0][col],
                                                                   peps.ltensors[1][col-1],
                                                                   None,
                                                                   step_size,
                                                                   ham[col])
        else:
            peps[col],peps.ltensors[0][col] = tebd_step_single_col(peps[col],
                                                                   peps.ltensors[0][col],
                                                                   peps.ltensors[1][col-1],
                                                                   peps.ltensors[1][col],
                                                                   step_size,
                                                                   ham[col])
    # Return result
    return peps

def tebd_step(peps,ham,step_size):
    """
    """
    # Columns ----------------------------------
    peps = tebd_step_col(peps,ham[0],step_size)
    # Rows -------------------------------------
    peps.rotate(clockwise=True)
    peps = tebd_step_col(peps,ham[1],step_size)
    peps.rotate(clockwise=False)
    # Return results ---------------------------
    return peps

def tebd_steps(peps,ham,step_size,n_step,conv_tol,chi=None):
    """
    """
    nSite = len(peps)*len(peps[0])

    # Compute Initial Energy
    Eprev = peps.calc_op(ham,chi=chi)
    mpiprint(0,'Initial Energy/site = {}'.format(Eprev/nSite))

    # Do a single tebd step
    for iter_cnt in range(n_step):

        # Do TEBD Step
        peps = tebd_step(peps,ham,step_size)

        # Normalize just in case
        peps.normalize()
        
        # Compute Resulting Energy
        E = peps.calc_op(ham,chi=chi)
        
        # Check for convergence
        mpiprint(0,'Energy/site = {} '.format(E/nSite))
        if abs((E-Eprev)/E) < conv_tol:
            mpiprint(3,'Converged E = {} to an accuracy of ~{}'.format(E,abs(E-Eprev)))
            converged = True
            break
        else:
            Eprev = E
            converged = False
    return E,peps

def run_tebd(Nx,Ny,d,ham,
             Zn=None,
             peps=None,
             backend='numpy',
             D=3,chi=10,
             norm_tol=20,singleLayer=True,
             max_norm_iter=20,
             dtype=float_,
             step_size=0.2,n_step=5,conv_tol=1e-8):
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
        Zn : int
            The Zn symmetry of the PEPS. 
            If None, then a dense, non-symmetric PEPS will be used.
        backend : str
            The tensor backend to be used. 
            Current options are 'numpy' or 'ctf'
        peps : PEPS object
            The initial guess for the PEPS, in the "Gamma-Lambda"
            formalism. If this is not 
            provided, then a random peps will be used. 
            Note that the bond dimension D should be the same
            as the initial calculation bond dimension, since
            no bond reduction or initial increase of bond dimension
            is currently implemented.
        D : int
            The maximum bond dimension (may be a list of
            maximum bond dimensions, and a loop will be
            performed where it is slowly incremented)
        chi : int
            The boundary mpo maximum bond dimension
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
        n_step : int
            The number of steps to be taken for each
            trotter step size. If it is a list, then 
            len(step_size) == len(n_step) and 
            len(D) == len(n_step) must both be True.
        conv_tol : float
            The convergence tolerance

    """
    t0 = time.time()
    mpiprint(0,'\n\nStarting SU TEBD Calculation')
    mpiprint(0,'#'*50)

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
        peps = PEPS(Nx=Nx,
                    Ny=Ny,
                    d=d,
                    D=D[0],
                    chi=chi[0],
                    Zn=Zn,
                    backend=backend,
                    norm_tol=norm_tol,
                    canonical=True,
                    singleLayer=singleLayer,
                    max_norm_iter=max_norm_iter,
                    dtype=dtype)
    
    # Loop over all (bond dims/step sizes/number of steps)
    for Dind in range(len(D)):

        mpiprint(0,'\nSU Calculation for (D,chi,dt) = ({},{},{})'.format(D[Dind],chi[Dind],step_size[Dind]))
        
        # Do a tebd evolution for given step size
        E,peps = tebd_steps(peps,
                            ham,
                            step_size[Dind],
                            n_step[Dind],
                            conv_tol[Dind],
                            chi = chi[Dind])

        # Increase MBD if needed
        if (len(D)-1 > Dind) and (D[Dind+1] > D[Dind]):
            peps.increase_mbd(D[Dind+1],chi=chi[Dind+1])

    # Print out results
    mpiprint(0,'\n\n'+'#'*50)
    mpiprint(0,'SU TEBD Complete')
    mpiprint(0,'-------------')
    mpiprint(0,'Total time = {} s'.format(time.time()-t0))
    mpiprint(0,'Per Site Energy = {}'.format(E/(Nx*Ny)))

    return E,peps

if __name__ == "__main__":
    # PEPS parameters
    Nx = 3
    Ny = 3
    d = 2
    D = 5
    chi = 10
    # Get Hamiltonian
    from cyclopeps.ops.itf import return_op
    ham = return_op(Nx,Ny,(1.,2.))
    # Run TEBD
    E,_ = run_tebd(Nx,Ny,d,ham,
                   D=[1,D,D,D,D],
                   chi=chi,
                   singleLayer=True,
                   max_norm_iter=100,
                   dtype=float_,
                   step_size=[1.,0.1,0.01,0.001,0.0001],
                   n_step=[10,10,10,10,10])
