from cyclopeps.tools.params import *
from cyclopeps.tools.utils import *
from cyclopeps.tools.mps_tools import *
from cyclopeps.tools.peps_tools import *
from cyclopeps.tools.ops_tools import *
from cyclopeps.algs.simple_update import run_tebd as su
from numpy import float_,isfinite
import numpy as np

#@profile
def cost_func(N, bot, top, bot_new, top_new, U, reduced=True):
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
    aRa = calc_local_op(bot_new, 
                        top_new, 
                        N, 
                        None, 
                        normalize=False, 
                        reduced=reduced)

    # Calculate a^+ * S
    # PH - Need to make sure the tensors aren't flipped for time evolution
    Sa = calc_local_op(bot_new,
                       top_new,
                       N,
                       U,
                       phys_b_ket=bot,
                       phys_t_ket=top,
                       normalize=False,
                       reduced=reduced)

    # Calculate S^+ * a
    # (currently just using a copy)
    aS  = Sa

    # Return Cost Function
    return aRa-aS-Sa

#@profile
def optimize_bottom(N, bot, top, bot_new, top_new, eH, reduced=True):
    """
    Note:
        implemented according to Section II.B of
        https://arxiv.org/pdf/1503.05345.pdf
    """
    if reduced:

        # Calculate R
        tmp = einsum('DPU,dPu->DUdu', top_new, top_new.copy().conj())
        R   = einsum('DUdu,AaUu->ADad', tmp, N)

        # Calculate S
        tmp = einsum('APD,AaBb->DPaBb', bot, N)
        tmp = einsum('DPaBb,DQB->PaQb', tmp, top)

        if len(tmp.legs[0]) == 2:
            # Then thermal state
            tmp.unmerge_ind(2)
            tmp.unmerge_ind(0)
            tmp = einsum('PxaQyb,PQpq->abpxqy', tmp, eH)
            tmp.merge_inds([4,5])
            tmp.merge_inds([2,3])
        else:
            # Then regular state
            tmp = einsum('PaQb,PQpq->abpq', tmp, eH)
        S = einsum('abpq,dqb->apd', tmp, top_new)

        # Take inverse of R
        R_ = R.square_inv()

        # Compute phys_b_new
        bot_new = einsum('ADad,apd->ApD', R_, S)

    else:

        # Calculate R
        tmp = einsum('KZQSU,kzQsu->KZSUkzsu', top_new, top_new.copy().conj())
        R   = einsum('KZSUkzsu,lLdDrRkKuUsS->LDRZldrz', tmp, N)

        # Calculate S
        tmp = einsum('LDPRZ,lLdDrRkKuUsS->PZldrkKuUsS', bot, N)
        tmp = einsum('PZldrkKuUsS,KZQSU->PQldrkus', tmp, top)

        if len(tmp.legs[0]) == 2:
            # Then thermal state
            tmp.unmerge_ind(1)
            tmp.unmerge_ind(0)
            tmp = einsum('PxQyldrkus,PQpq->pxqyldrkus', tmp, eH)
            tmp.merge_inds([0,1])
            tmp.merge_inds([1,2])
        else:
            # Then regular state
            tmp = einsum('PQldrkus,PQpq->pqldrkus', tmp, eH)
        S = einsum('pqldrkus,kzqsu->ldprz', tmp, top_new)

        # Take inverse of R
        R_ = R.square_inv()

        # Compute phys_b_new
        bot_new = einsum('LDRZldrz,ldprz->LDpRZ', R_, S)

    # Return Results
    return bot_new

#@profile
def optimize_top(N, bot, top, bot_new, top_new, eH, use_inv=False, reduced=True):
    """
    Note:
        implemented according to Section II.B of
        https://arxiv.org/pdf/1503.05345.pdf
    """
    if reduced:

        # Calculate R
        tmp = einsum('DPU,dPu->DUdu', bot_new, bot_new.copy().conj())
        R = einsum('DUdu,DdAa->UAua', tmp, N)

        # Calculate S
        tmp = einsum('UQA,DdAa->UQDda', top, N)
        tmp = einsum('UQDda,DPU->PQad', tmp, bot)
        if len(tmp.legs[0]) == 2:
            tmp.unmerge_ind(1)
            tmp.unmerge_ind(0)
            tmp = einsum('PxQyad,PQpq->pxqyad', tmp, eH)
            tmp.merge_inds([2,3])
            tmp.merge_inds([0,1])
        else:
            tmp = einsum('PQad,PQpq->pqad', tmp, eH)
        S   = einsum('pqad,dpu->uqa', tmp, bot_new)

        # Solve least squares problem
        # Take inverse of R
        R_ = R.square_inv()

        # Compute new phys_t_new
        phys_t_new = einsum('uaUA,uqa->UqA', R_, S)

    else:

        # Calculate R
        tmp = einsum('LDPRZ,ldPrz->LDRZldrz', bot_new, bot_new.copy().conj())
        R = einsum('LDRZldrz,lLdDrRkKuUsS->KZSUkzsu', tmp, N)

        # Calculate S
        tmp = einsum('KZQSU,lLdDrRkKuUsS->ZQlLdDrRkus', top, N)
        tmp = einsum('ZQlLdDrRkus,LDPRZ->PQldrkus', tmp, bot)
        if len(tmp.legs[0]) == 2:
            tmp.unmerge_ind(1)
            tmp.unmerge_ind(0)
            tmp = einsum('PxQyldrkus,PQpq->pxqyldrkus', tmp, eH)
            tmp.merge_inds([2,3])
            tmp.merge_inds([0,1])
        else:
            tmp = einsum('PQldrkus,PQpq->pqldrkus', tmp, eH)
        S   = einsum('pqldrkus,ldprz->kzqsu', tmp, bot_new)

        # Solve least squares problem
        # Take inverse of R
        R_ = R.square_inv()

        # Compute new phys_t_new
        top_new = einsum('kzsuKZSU,kzqsu->KZqSU', R_, S)

    # Return Results
    return top_new

def split_sites(comb, mbd, reduced=True):
    """
    Given two combined (reduced) sites,
    split them back into two separated
    tensor sites via svd
    """
    if reduced:
        # Do the SVD Decomposition
        site1,sings,site2 = comb.svd(2,
                                     truncate_mbd=mbd,
                                     return_ent=False,
                                     return_wgt=False)

        # Do some renormalization
        sings /= einsum('ij,jk->ik', sings, sings).sqrt().to_val()

        # Absorb singular values into sites
        site1 = einsum('ijk,kl->ijl', site1, sings.sqrt())
        site2 = einsum('ij,jkl->ikl', sings.sqrt(), site2)
    else:
        # Do the SVD Decomposition
        site1,sings,site2 = comb.svd(4,
                                     truncate_mbd=mbd,
                                     return_ent=False,
                                     return_wgt=False)

        # Do some renormalization
        sings /= einsum('ij,jk->ik', sings, sings).sqrt().to_val()

        # Absorb singular values into sites
        site1 = einsum('LDPRX,XU->LDPRU', site1, sings.sqrt())
        site2 = einsum('DX,XLPRU->LDPRU', sings.sqrt(), site2)

    # Return the result
    return site1,site2

def simple_update_init_guess_reduced(bot, top, eH, mbd):
    """
    Create an initial guess for the ALS procedure 
    via a simple update style update, for the 
    reduced peps tensors (meaning the physical index
    has been pulled off the peps tensor via a qr)
    """
    # Apply time evolution gate
    comb = einsum('DPA,AQU->DPQU',bot,top)
    if len(comb.legs[1]) == 2:
        # Thermal State time evolution
        comb.unmerge_ind(2)
        comb.unmerge_ind(1)
        comb = einsum('DPxQyU,PQpq->DpxqyU',comb,eH)
        comb.merge_inds([1,2])
        comb.merge_inds([2,3])
    else:
        # Regular state time evolution
        comb = einsum('DPQU,PQpq->DpqU',comb,eH)

    # Split via SVD
    bot, top = split_sites(comb, mbd, reduced=True)

    # Return Results
    return bot, top

def simple_update_init_guess_full(bot, top, eH, mbd):
    """
    Create an initial guess for the ALS procedure 
    via a simple update style update, for the 
    full peps tensors (rank 5)
    """
    # Apply time evolution gate
    comb = einsum('LDPRZ,lZQru->LDPRlQru',bot,top)

    if len(comb.legs[2]) == 2:
        # Thermal State Time Evolution
        comb.unmerge_ind(5)
        comb.unmerge_ind(2)
        comb = einsum('LDPxRlQyru,PQpq->LDpxRlqyru',comb,eH)
        comb.merge_inds([2,3])
        comb.merge_inds([5,6])
    else:
        # Regurlar state time evolution
        comb = einsum('LDPRlQru,PQpq->LDpRlqru',comb,eH)

    # Split via SVD
    bot, top = split_sites(comb, mbd, reduced=False)

    # Return Results
    return bot, top

def simple_update_init_guess(bot, top, eH, mbd, reduced=True):
    """
    Create an initial guess for the ALS procedure
    via a simple update style update
    """
    if reduced:
        return simple_update_init_guess_reduced(bot,
                                                top,
                                                eH,
                                                mbd)
    else:
        return simple_update_init_guess_full(bot,
                                             top,
                                             eH,
                                             mbd)

#@profile
def noiseless_als(bot, top, N, eH, mbd,
                  als_iter=100, als_tol=1e-10, stablesplit=True,
                  use_su=True, reduced=True):
    """
    Do the Alternating least squares procedure, using the current
    physical index-holding tensors as the initial guess
    """
    # Create initial guesses for resulting tensors
    # #print('We should just add noise here')

    if use_su:
        bot_new, top_new = simple_update_init_guess(bot, top, eH, mbd, reduced=reduced)
    else:
        bot_new, top_new = bot.copy(), top.copy()

    # Initialize cost function
    cost_prev = cost_func(N,
                          bot,
                          top,
                          bot_new,
                          top_new,
                          eH,
                          reduced=reduced)

    for i in range(als_iter):
        #print('\t\t\t\tALS iter {}'.format(i))

        # Optimize Bottom Site
        #print('\t\t\t\t\tOptimizing bottom')
        bot_new = optimize_bottom(N,
                                  bot,
                                  top,
                                  bot_new,
                                  top_new,
                                  eH,
                                  reduced=reduced)

        # Optimize Top Site
        #print('\t\t\t\t\tOptimizing top')
        top_new = optimize_top(N,
                               bot,
                               top,
                               bot_new,
                               top_new,
                               eH,
                               reduced=reduced)
        # Split singular values to stabilize
        if stablesplit:
            if reduced:
                comb = einsum('DPA,AQU->DPQU', bot_new, top_new)
            else:
                comb = einsum('LDPRZ,KZQSU->LDPRKQSU', bot_new, top_new)
            bot_new, top_new = split_sites(comb, mbd, reduced=reduced)

        # Compute cost function 
        #print('\t\t\t\t\tConst Function')
        cost = cost_func(N,
                         bot,
                         top,
                         bot_new,
                         top_new,
                         eH,
                         reduced=reduced)

        # Check for convergence
        if (abs(cost) < als_tol) or (abs((cost-cost_prev)/cost) < als_tol):
            break
        else:
            cost_prev = cost

    # Return result
    return bot_new, top_new

def alternating_least_squares(bot, top, N, eH, mbd,
                              als_iter=100, als_tol=1e-10,
                              reduced=True):
    """
    Do alternating least squares to determine best tensors
    to represent time evolved tensors at smaller bond dimensions
    """
    # Run ALS
    res = noiseless_als(bot,
                        top,
                        N,
                        eH,
                        mbd,
                        als_iter=als_iter,
                        als_tol=als_tol,
                        reduced=reduced)

    # Return result
    return res

#@profile
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

def tebd_single_gate(row, ham, peps_col, left_bmpo, right_bmpo,
                     top_envs, bot_envs, step_size, mbd,
                     als_iter=100, als_tol=1e-10, in_mem=True,
                     reduced=True):
    """
    """
    # Calculate environment aroudn reduced tensors
    #print('\t\t\tCalculating Env')
    res = calc_N(row,
                 peps_col,
                 left_bmpo,
                 right_bmpo,
                 top_envs,
                 bot_envs,
                 in_mem=in_mem,
                 reduced=reduced)
    if reduced:
        peps_b,bot,top,peps_t,_,_,_,_,N = res
    else:
        bot = peps_col[row]
        top = peps_col[row+1]
        N   = res

    # Take the exponential of the hamiltonian
    #print('\t\t\tExponentiating gate')
    eH = exp_gate(ham[row],-step_size)

    # Do alternating least squares to find new peps tensors
    #print('\t\t\tALS')
    bot,top = alternating_least_squares(bot,
                                        top,
                                        N,
                                        eH,
                                        mbd,
                                        als_iter=als_iter,
                                        als_tol=als_tol,
                                        reduced=reduced)

    # Calculate Energy & Norm
    #print('\t\t\tCalc local op')
    E,norm = calc_local_op(bot,
                           top,
                           N,
                           ham[row],
                           return_norm=True,
                           reduced=reduced)

    # Update peps_col tensors
    if reduced:
        peps_col[row]   = einsum('LDRa,aPU->LDPRU', peps_b,    bot)
        peps_col[row+1] = einsum('DPa,LaRU->LDPRU',    top, peps_t)
    else:
        peps_col[row]   = bot
        peps_col[row+1] = top

    # Combine and equally split the two tensors
    peps_col[row],peps_col[row+1] = make_equal_distance(peps_col[row],peps_col[row+1],mbd)

    # Update top and bottom environments
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
    norm_fact = bot_envs[row].abs().max()
    bot_envs[row] /= norm_fact
    peps_col[row] /= norm_fact**(1./2.)
    peps_col[row+1] /= norm_fact**(1./2.)

    # Write everything back to disk (if wanted)
    if not in_mem:
        bot_envs[row].to_disk()
        peps_col[row].to_disk()
        peps_col[row+1].to_disk()

    return E, peps_col, bot_envs, top_envs

#@profile
def tebd_step_single_col(peps_col, step_size, left_bmpo, right_bmpo, ham, mbd,
                         als_iter=100, als_tol=1e-10, in_mem=True, reduced=True):
    """
    """
    # Calculate top and bottom environments
    top_envs = calc_top_envs(peps_col,left_bmpo,right_bmpo,in_mem=in_mem)
    bot_envs = calc_bot_envs(peps_col,left_bmpo,right_bmpo,in_mem=in_mem)

    # Set up array to store energies of sites in columns
    E = peps_col[0].backend.zeros(len(ham),dtype=peps_col[0].dtype)

    # Loop through rows in the column
    for row in range(len(ham)):
        #print('\t\tDoing tebd step on site {} out of {}'.format(row,len(ham)))

        # Do TEBD on a single duo of sites
        out = tebd_single_gate(row,
                               ham,
                               peps_col,
                               left_bmpo,
                               right_bmpo,
                               top_envs,
                               bot_envs,
                               step_size,
                               mbd,
                               als_iter=als_iter,
                               als_tol=als_tol,
                               in_mem=in_mem,
                               reduced=reduced)

        # Unpack output
        E[row]   = out[0]
        peps_col = out[1]
        bot_envs = out[2]
        top_envs = out[3]

    # Return the result
    return E,peps_col

#@profile
def tebd_step_col(peps,ham,step_size,mbd,
                  chi=None,als_iter=100,
                  als_tol=1e-10,in_mem=True,
                  reduced=True):
    """
    """
    # Figure out peps size
    (Nx,Ny) = peps.shape

    # Compute the boundary MPOs
    right_bmpo = calc_right_bound_mpo(peps, 0,
                                      chi=chi,
                                      in_mem=in_mem,
                                      return_all=True)
    left_bmpo  = [None]*(Nx-1)
    ident_bmpo = identity_mps(len(right_bmpo[0]),
                              dtype=peps[0][0].dtype,
                              sym=(peps[0][0].sym is not None),
                              backend=peps.backend)

    # Loop through all columns
    E = peps.backend.zeros((len(ham),len(ham[0])),dtype=peps[0][0].dtype)
    for col in range(Nx):

        # Take TEBD Step
        #print('\tDoing tebd step on col/row {} out of {}'.format(col,Nx))
        if col == 0:
            res = tebd_step_single_col(peps[col],
                                       step_size,
                                       ident_bmpo,
                                       right_bmpo[col],
                                       ham[col],
                                       mbd,
                                       als_iter=als_iter,
                                       als_tol=als_tol,
                                       in_mem=in_mem,
                                       reduced=reduced)
        elif col == Nx-1:
            res = tebd_step_single_col(peps[col],
                                       step_size,
                                       left_bmpo[col-1],
                                       ident_bmpo,
                                       ham[col],
                                       mbd,
                                       als_iter=als_iter,
                                       als_tol=als_tol,
                                       in_mem=in_mem,
                                       reduced=reduced)
        else:
            res = tebd_step_single_col(peps[col],
                                       step_size,
                                       left_bmpo[col-1],
                                       right_bmpo[col],
                                       ham[col],
                                       mbd,
                                       als_iter=als_iter,
                                       als_tol=als_tol,
                                       in_mem=in_mem,
                                       reduced=reduced)
        E[col,:] = res[0]
        peps[col] = res[1]

        # Ensure Needed PEPS & BMPO columns are in memory
        if not in_mem:
            peps.col_from_disk(col)
            if col > 0:
                left_bmpo[col-1].from_disk()

        # Update left boundary tensors
        if col == 0:
            left_bmpo[col] = update_left_bound_mpo(peps[col], None, chi=chi)
        elif col != Nx-1:
            left_bmpo[col] = update_left_bound_mpo(peps[col], left_bmpo[col-1], chi=chi)

        # Write everything in memory to disk (if needed)
        if not in_mem:
            peps.col_to_disk(col)
            if col > 0:
                left_bmpo[col-1].to_disk()
            elif col != Nx-1:
                left_bmpo[col].to_disk()

    # Return result
    return E,peps

#@profile
def tebd_step(peps,ham,step_size,mbd,
              chi=None,als_iter=100,als_tol=1e-10,
              print_prepend='',in_mem=True,reduced=True):
    """
    """
    # Columns ----------------------------------
    #print('Doing TEBD step on columns')
    Ecol,peps = tebd_step_col(peps,ham[0],step_size,mbd,
                              chi=chi,
                              als_iter=als_iter,
                              als_tol=als_tol,
                              in_mem=in_mem,
                              reduced=reduced)

    # Rows -------------------------------------
    #print('Doing TEBD step on rows')
    peps.rotate(clockwise=True)
    Erow,peps = tebd_step_col(peps,ham[1],step_size,mbd,
                              chi=chi,
                              als_iter=als_iter,
                              als_tol=als_tol,
                              in_mem=in_mem,
                              reduced=reduced)
    peps.rotate(clockwise=False)

    # Return results ---------------------------
    mpiprint(5,print_prepend+'Column energies =\n{}'.format(Ecol))
    mpiprint(5,print_prepend+'Row energies =\n{}'.format(Erow))
    # Sum energies
    E = peps.backend.sum(Ecol)+peps.backend.sum(Erow)

    return E,peps

def tebd_steps(peps,ham,step_size,n_step,conv_tol,mbd,
               chi=None,chi_norm=10,chi_op=None,
               als_iter=100,als_tol=1e-10,
               print_prepend='',save_all_steps=False,
               in_mem=True,reduced=True):
    """
    """
    nSite = len(peps)*len(peps[0])

    # Compute Initial Energy
    mpiprint(3,print_prepend+'Calculation Initial Energy/site')
    Eprev = peps.calc_op(ham,
                         chi=chi_op,
                         in_mem=in_mem)
    mpiprint(0,print_prepend+'Initial Energy/site = {}'.format(Eprev/nSite))

    # Do a single tebd step
    for iter_cnt in range(n_step):

        # Do TEBD Step
        _,peps = tebd_step(peps,ham,step_size,mbd,
                           chi=chi,
                           als_iter=als_iter,
                           als_tol=als_tol,
                           print_prepend=print_prepend,
                           in_mem=in_mem,
                           reduced=reduced)

        # Normalize just in case
        peps.normalize(chi=chi_norm, in_mem=in_mem)

        # Save PEPS
        if save_all_steps: 
            peps.save(fname=peps.fname+'_iter{}'.format(iter_cnt))
        else:
            peps.save()
        
        # Compute Resulting Energy
        E = peps.calc_op(ham,
                         chi=chi_op,
                         in_mem=in_mem)

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
             reduced=True,
             peps_fname=None,
             peps_fdir='./',
             in_mem=True,
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
        reduced : bool
            Whether or not to use the reduced update (True) 
            where the physical index is pulled off the peps
            tensor via a QR before the ALS procedure is done
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
        in_mem : bool
            Whether all arrays will be stored in local memory
            or written to disk (default is False). The location
            of the directory where this is stored is specified 
            by the environment variable TMPDIR, where a randomly 
            named directory will be created and all temporary files
            will be stored there.
        save_all_steps : bool
            Whether to save all states of the PEPS 
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
                        fdir=peps_fdir,
                        in_mem=in_mem)
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
                        # TODO - Implement SU out of memory
                        #in_mem=in_mem)
            peps.absorb_lambdas()

    # Absorb lambda tensors if canonical
    if peps.ltensors is not None:
        peps.absorb_lambdas()

    # Push PEPS to disk (if not already)
    if not in_mem:
        peps.to_disk()
    
    # Make sure the peps is normalized
    peps.normalize(chi=chi_norm,in_mem=in_mem)

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
                            save_all_steps=save_all_steps,
                            in_mem=in_mem,
                            reduced=reduced)

    # Print out results
    mpiprint(0,'\n\n'+print_prepend+'#'*50)
    mpiprint(0,print_prepend+'FU TEBD Complete')
    mpiprint(0,print_prepend+'-------------')
    mpiprint(0,print_prepend+'Total time = {} s'.format(time.time()-t0))
    mpiprint(0,print_prepend+'Per Site Energy = {}'.format(E/(Nx*Ny)))

    return E,peps
