from cyclopeps.tools.utils import *
from cyclopeps.tools.peps_tools import *
from cyclopeps.tools.ops_tools import *
from numpy import float_

def absorb_lambdas(row,peps_col,vert_lambdas,left_lambdas,right_lambdas):
    """
    """
    _peps1 = peps_col[row].copy().make_sparse()
    _peps2 = peps_col[row+1].copy().make_sparse()
    # Do Dense version ----------------------------------------------------------------
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
    # Do Sparse version ----------------------------------------------------------------
    # Absorb Bottom lambda
    if not (row == 0):
        _peps1 = einsum('ldpru,dD->lDpru',_peps1,vert_lambdas[row-1].make_sparse())
    # Absorb left lambdas
    if left_lambdas is not None:
        _peps1 = einsum('ldpru,lL->Ldpru',_peps1,left_lambdas[row].make_sparse())
        _peps2 = einsum('ldpru,lL->Ldpru',_peps2,left_lambdas[row+1].make_sparse())
    # Absorb right lambdas
    if right_lambdas is not None:
        _peps1 = einsum('ldpru,rR->ldpRu',_peps1,right_lambdas[row].make_sparse())
        _peps2 = einsum('ldpru,rR->ldpRu',_peps2,right_lambdas[row+1].make_sparse())
    # Absorb Top lambda
    if not (row == len(peps_col)-2):
        _peps2 = einsum('ldpru,uU->ldprU',_peps2,vert_lambdas[row+1].make_sparse())
    # Absorb middle lambda
    _peps1 = einsum('ldpru,uU->ldprU',_peps1,vert_lambdas[row].make_sparse())
    # Compare results -------------------------------------------------------------------
    #print('PEPS 1 diff (absorb): {}'.format((_peps1-peps1.make_sparse()).abs().sum()))
    #print('PEPS 2 diff (absorb): {}'.format((_peps2-peps2.make_sparse()).abs().sum()))
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
    _peps1 = peps_col[row].copy().make_sparse()
    _peps2 = peps_col[row+1].copy().make_sparse()
    # Compare invert_diag results
    v1 = vert_lambdas[row-1].copy().invert_diag().make_sparse()
    v2 = vert_lambdas[row-1].copy().make_sparse().invert_diag()
    # Do Dense version ----------------------------------------------------------------
    peps1 = peps_col[row].copy()
    peps2 = peps_col[row+1].copy()
    # Absorb Bottom lambda
    if not (row == 0):
        peps1 = einsum('ldpru,dD->lDpru',peps1,vert_lambdas[row-1].invert_diag())
    # Absorb left lambdas
    if left_lambdas is not None:
        peps1 = einsum('ldpru,lL->Ldpru',peps1,left_lambdas[row].invert_diag())
        peps2 = einsum('ldpru,lL->Ldpru',peps2,left_lambdas[row+1].invert_diag())
    # Absorb right lambdas
    if right_lambdas is not None:
        peps1 = einsum('ldpru,rR->ldpRu',peps1,right_lambdas[row].invert_diag())
        peps2 = einsum('ldpru,rR->ldpRu',peps2,right_lambdas[row+1].invert_diag())
    # Absorb Top lambda
    if not (row == len(peps_col)-2):
        peps2 = einsum('ldpru,uU->ldprU',peps2,vert_lambdas[row+1].invert_diag())

    # Put them back in the list
    peps_col[row] = peps1
    peps_col[row+1] = peps2
    # Do Sparse version ----------------------------------------------------------------
    # Absorb Bottom lambda
    if not (row == 0):
        _peps1 = einsum('ldpru,dD->lDpru',_peps1,vert_lambdas[row-1].make_sparse().invert_diag())
    # Absorb left lambdas
    if left_lambdas is not None:
        _peps1 = einsum('ldpru,lL->Ldpru',_peps1,left_lambdas[row].make_sparse().invert_diag())
        _peps2 = einsum('ldpru,lL->Ldpru',_peps2,left_lambdas[row+1].make_sparse().invert_diag())
    # Absorb right lambdas
    if right_lambdas is not None:
        _peps1 = einsum('ldpru,rR->ldpRu',_peps1,right_lambdas[row].make_sparse().invert_diag())
        _peps2 = einsum('ldpru,rR->ldpRu',_peps2,right_lambdas[row+1].make_sparse().invert_diag())
    # Absorb Top lambda
    if not (row == len(peps_col)-2):
        _peps2 = einsum('ldpru,uU->ldprU',_peps2,vert_lambdas[row+1].make_sparse().invert_diag())
    #print('PEPS 1 diff (remove): {}'.format((_peps1-peps1.make_sparse()).abs().sum()))
    #print('PEPS 2 diff (remove): {}'.format((_peps2-peps2.make_sparse()).abs().sum()))

    # Return result
    return peps_col


def tebd_step_single_col(peps_col,vert_lambdas,left_lambdas,right_lambdas,step_size,ham):
    """
    """

    # Loop through rows in the column
    E = zeros(len(ham),dtype=peps_col[0].dtype)
    for row in range(len(ham)):

        # Get symmetries for reference
        sym1,sym2 = peps_col[row].get_signs(), peps_col[row+1].get_signs()

        # Absorb Lambdas into Gamma tensors
        peps1,peps2 = absorb_lambdas(row,peps_col,vert_lambdas,left_lambdas,right_lambdas)

        # Take the exponential of the hamiltonian
        eH = exp_gate(ham[row],-step_size)

        # Apply Time Evolution
        tmp = einsum('ldpru,LuPRU->ldprLPRU',peps1,peps2)
        result = einsum('ldprLPRU,pPqQ->ldqrLQRU',tmp,eH)

        # Perform SVD
        D = peps1.shape[4]
        peps1,Lambda,peps2 = separate_sites(result,D)
        comparison_ten = einsum('ldpru,LuPRU->ldprLPRU',einsum('ldpru,uU->ldprU',peps1,Lambda),peps2).make_sparse()

        # Put result back into vectors
        vert_lambdas[row] = Lambda
        peps_col[row]   = peps1
        peps_col[row+1] = peps2
        comparison2_ten = einsum('ldpru,LuPRU->ldprLPRU',einsum('ldpru,uU->ldprU',peps_col[row],vert_lambdas[row]),peps_col[row+1]).make_sparse()
        #print('diff1 = {}'.format((comparison_ten-comparison2_ten).abs().sum()))

        # Remove Lambdas
        peps_col = remove_lambdas(row,peps_col,vert_lambdas,left_lambdas,right_lambdas)

        # Update symmetries
        peps_col[row].update_signs(sym1)
        peps_col[row+1].update_signs(sym2)

        # A final check
        peps1,peps2 = absorb_lambdas(row,peps_col,vert_lambdas,left_lambdas,right_lambdas)
        comparison3_ten = einsum('ldpru,LuPRU->ldprLPRU',peps1,peps2).make_sparse()
        #print('diff2 = {}'.format((comparison_ten-comparison3_ten).abs().sum()))

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

        # Save PEPS
        #peps.save()

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
             D=3,
             chi=10,
             norm_tol=20,
             singleLayer=True,
             max_norm_iter=20,
             dtype=float_,
             step_size=0.2,
             n_step=5,
             conv_tol=1e-8,
             peps_fname=None,
             peps_fdir='./'):
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
        peps_fname : str
            The name of the saved peps file
        peps_fdir : str
            The location where the peps will be saved
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
                    dtype=dtype,
                    fname=peps_fname,
                    fdir=peps_fdir)
    
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
            peps.normalize()

    # Print out results
    mpiprint(0,'\n\n'+'#'*50)
    mpiprint(0,'SU TEBD Complete')
    mpiprint(0,'-------------')
    mpiprint(0,'Total time = {} s'.format(time.time()-t0))
    mpiprint(0,'Per Site Energy = {}'.format(E/(Nx*Ny)))

    return E,peps
