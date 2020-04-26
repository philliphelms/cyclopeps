from cyclopeps.tools.params import *
from cyclopeps.tools.utils import *
from cyclopeps.tools.mps_tools import *
from cyclopeps.tools.peps_tools import *
from cyclopeps.tools.ops_tools import *
from cyclopeps.tools.gen_ten import GEN_TEN,rand
from cyclopeps.algs.simple_update import run_tebd as su
from numpy import float_,isfinite

def cost_func(bra,eH,top,bot,left,right,u,v,p,q,truncate=False,chi=100):
    """
    Compute <psi_t|psi_t>-2*<psi_t|psi_p>
    which should be zero for a perfect fit
    """
    # Check if dealing with a thermal state
    thermal = len(bra[0][0].legs[2]) == 2
    # Contract state with time evolution operator
    Hbra = [[None,None],[None,None]]
    if thermal:
        bra[0][1].unmerge_ind(2)
        Hbra[0][1] = einsum('ldparu,pPx->ldxParu',bra[0][1],eH[0])
        Hbra[0][1].merge_inds([3,4])
        bra[0][1].merge_inds([2,3])
        bra[0][0].unmerge_ind(2)
        Hbra[0][0] = einsum('ldparu,xpPy->ldParyux',bra[0][0],eH[1])
        Hbra[0][0].merge_inds([2,3])
        bra[0][0].merge_inds([2,3])
        bra[1][0].unmerge_ind(2)
        Hbra[1][0] = einsum('ldparu,ypP->lydParu',bra[1][0],eH[2])
        Hbra[1][0].merge_inds([3,4])
        bra[1][0].merge_inds([2,3])
        Hbra[1][1] = bra[1][1].copy()
    else:
        Hbra[0][1] = einsum('ldpru,pPx->ldxPru',bra[0][1],eH[0]) # Top left site
        Hbra[0][0] = einsum('ldpru,xpPy->ldPryux',bra[0][0],eH[1]) # Bottom left site
        Hbra[1][0] = einsum('ldpru,ypP->lydPru',bra[1][0],eH[2]) # Bottom right site
        Hbra[1][1] = bra[1][1].copy()
    # Merge indices as needed
    Hbra[0][1].merge_inds([1,2])
    Hbra[0][0].merge_inds([3,4])
    Hbra[0][0].merge_inds([4,5])
    Hbra[1][0].merge_inds([0,1])
    # Contract with reducing tensors -----------------
    Hbra_red = [[None,None],[None,None]]
    Hbra_red[0][1] = einsum('ldpru,dD->lDpru',Hbra[0][1],u)
    Hbra_red[0][0] = einsum('ldpru,rR->ldpRu',Hbra[0][0],p)
    Hbra_red[0][0] = einsum('ldpru,uU->ldprU',Hbra_red[0][0],v)
    Hbra_red[1][0] = einsum('ldpru,lL->Ldpru',Hbra[1][0],q)
    Hbra_red[1][1] = Hbra[1][1].copy()

    # Contract <psi_t|psi_t>
    ttbot = update_bot_env2(0,
                           Hbra_red,
                           Hbra_red,
                           left[0],
                           left[1],
                           right[0],
                           right[1],
                           bot,
                           truncate=truncate,
                           chi=chi)
    tttop = update_top_env2(1,
                           Hbra_red,
                           Hbra_red,
                           left[2],
                           left[3],
                           right[2],
                           right[3],
                           top,
                           truncate=truncate,
                           chi=chi)
    tt = ttbot.contract(tttop)
    # Contracte <psi_t|psi_p>
    tpbot = update_bot_env2(0,
                           Hbra_red,
                           Hbra,
                           left[0],
                           left[1],
                           right[0],
                           right[1],
                           bot,
                           truncate=truncate,
                           chi=chi)
    tptop = update_top_env2(1,
                           Hbra_red,
                           Hbra,
                           left[2],
                           left[3],
                           right[2],
                           right[3],
                           top,
                           truncate=truncate,
                           chi=chi)
    tp = tpbot.contract(tptop)

    return 1.+tt-2.*tp

def rand_init_als_guess(Dpeps,Dham,chi):
    """
    Generate an  initial guess for the u,v,p, and q tensors
    """
    ten = GEN_TEN(shape=(Dpeps,Dham,chi))
    ten.randomize()
    ten.merge_inds([0,1])
    return ten

def su_init_als_guess_lb(bra,eH,mbd,add_noise=False):
    """
    Determine an initial guess for the ALS procedure
    using a simple splitting of tensors with svd

    Args:
        bra : list of two lists of length 2
            The peps tensors in the unit cell with
            peps[0][0] -> bottom left tensor
            peps[0][1] -> top left tensor
            peps[1][0] -> bottom right tensor
            peps[1][1] -> top right tensor
        eH : list of three tensors
            The exponentiated MPO for the three sites 
        mbd : int
            The maximum bond dimension for the PEPS

    Kwargs:
        add_noise : bool
            Whether to add noise into the initial guess
            when created this way. Default is False

    Returns:
        bra : list of two lists of length 2
            The time evolved tensors in the unit cell
            with time evolution done via application 
            of the exponentiated MPO followed by SVD 
            with truncation of small singular values
        
    """
    mpiprint(2,"\t\tGenerating initial ALS Guess")
    lib = bra[0][0].backend
    # Check if dealing with a thermal state
    thermal = len(bra[0][0].legs[2]) == 2
    # Contract state with time evolution operator
    Hbra = [[None,None],[None,None]]
    if thermal:
        bra[0][1].unmerge_ind(2)
        Hbra[0][1] = einsum('ldparu,pPx->ldxParu',bra[0][1],eH[0])
        Hbra[0][1].merge_inds([3,4])
        bra[0][1].merge_inds([2,3])
        bra[0][0].unmerge_ind(2)
        Hbra[0][0] = einsum('ldparu,xpPy->ldParyux',bra[0][0],eH[1])
        Hbra[0][0].merge_inds([2,3])
        bra[0][0].merge_inds([2,3])
        bra[1][0].unmerge_ind(2)
        Hbra[1][0] = einsum('ldparu,ypP->lydParu',bra[1][0],eH[2])
        Hbra[1][0].merge_inds([3,4])
        bra[1][0].merge_inds([2,3])
        Hbra[1][1] = bra[1][1].copy()
    else:
        Hbra[0][1] = einsum('ldpru,pPx->ldxPru',bra[0][1],eH[0]) # Top left site
        Hbra[0][0] = einsum('ldpru,xpPy->ldPryux',bra[0][0],eH[1]) # Bottom left site
        Hbra[1][0] = einsum('ldpru,ypP->lydPru',bra[1][0],eH[2]) # Bottom right site
        Hbra[1][1] = bra[1][1].copy()
    # Merge indices as needed
    Hbra[0][1].merge_inds([1,2])
    Hbra[0][0].merge_inds([3,4])
    Hbra[0][0].merge_inds([4,5])
    Hbra[1][0].merge_inds([0,1])

    # Create a reduced version -----------------------
    Hbra_red = [[None,None],[None,None]]
    Hbra_red[0][1] = Hbra[0][1].copy()
    Hbra_red[0][0] = Hbra[0][0].copy()
    Hbra_red[1][0] = Hbra[1][0].copy()
    Hbra_red[1][1] = Hbra[1][1].copy()

    # Shrink vertical bond --------------------------- 
    Hbra_red[0][1] = Hbra_red[0][1].transpose([0,2,3,4,1])
    U,S,V = Hbra_red[0][1].svd(4,truncate_mbd=mbd,return_ent=False,return_wgt=False)
    Hbra_red[0][1] = einsum('lprud,dD->lDpru',U,S.sqrt())
    gauge = einsum('ab,bc->ac',S.sqrt(),V)
    Hbra_red[0][0] = einsum('ldpru,Uu->ldprU',Hbra_red[0][0],gauge)
    if add_noise:
        noise = rand((Hbra_red[0][1].shape[1],Hbra_red[0][1].shape[1]),backend=lib)
        _noise = noise.copy()
        _noise.ten = lib.inv(noise.ten)
        Hbra_red[0][1] = einsum('ldpru,dD->lDpru',Hbra_red[0][1],noise)
        Hbra_red[0][0] = einsum('ldpru,Uu->ldprU',Hbra_red[0][0],_noise)

    # Determine u ------------------------------------
    H01 = Hbra[0][1].ten.copy()
    if thermal:
        H01 = H01.transpose([0,3,4,5,6,1,2])
        H01mat = H01.reshape((np.prod(H01.shape[:5]),np.prod(H01.shape[5:])))
        _H01mat = lib.pinv(H01mat)
        _H01 = _H01mat.reshape(H01.shape[5:]+H01.shape[:5])
        _H01 = GEN_TEN(ten=_H01)
        _H01.merge_inds([3,4])
    else:
        H01 = H01.transpose([0,3,4,5,1,2])
        H01mat = H01.reshape((np.prod(H01.shape[:4]),np.prod(H01.shape[4:])))
        _H01mat = lib.pinv(H01mat)
        _H01 = _H01mat.reshape(H01.shape[4:]+H01.shape[:4])
        _H01 = GEN_TEN(ten=_H01)
    u = einsum('ldpru,DXlpru->DXd',Hbra_red[0][1],_H01)
    u.merge_inds([0,1])
    tmp = einsum('Dd,lDpru->ldpru',u,Hbra[0][1])

    # Determine v ------------------------------------
    H00 = Hbra[0][0].ten.copy()
    if thermal:
        H00mat = H00.reshape((np.prod(H00.shape[:6]),np.prod(H00.shape[6:])))
        _H00mat = lib.pinv(H00mat)
        _H00 = _H00mat.reshape(H00.shape[6:]+H00.shape[:6])
        _H00 = GEN_TEN(ten=_H00)
        _H00.merge_inds([4,5])
    else:
        H00mat = H00.reshape((np.prod(H00.shape[:5]),np.prod(H00.shape[5:])))
        _H00mat = lib.pinv(H00mat)
        _H00 = _H00mat.reshape(H00.shape[5:]+H00.shape[:5])
        _H00 = GEN_TEN(ten=_H00)
    _H00.merge_inds([5,6])
    v = einsum('ldpru,UXldpr->UXu',Hbra_red[0][0],_H00)
    v.merge_inds([0,1])
    Hbra[0][0] = einsum('Uu,ldprU->ldpru',v,Hbra[0][0])

    # Shrink Horizontal Bond -------------------------
    U,S,V = Hbra_red[1][0].svd(1,truncate_mbd=mbd,return_ent=False,return_wgt=False)
    Hbra_red[1][0] = einsum('Ll,ldpru->Ldpru',S.sqrt(),V)
    gauge = einsum('ab,bc->ac',U,S.sqrt())
    Hbra_red[0][0] = einsum('ldpru,rR->ldpRu',Hbra_red[0][0],gauge)
    if add_noise:
        noise = rand((Hbra_red[1][0].shape[0],Hbra_red[1][0].shape[0]),backend=lib)
        _noise = noise.copy()
        _noise.ten = lib.inv(noise.ten)
        Hbra_red[1][0] = einsum('ldpru,Ll->Ldpru',Hbra_red[1][0],_noise)
        Hbra_red[0][0] = einsum('ldpru,rR->ldpRu',Hbra_red[0][0],noise)

    # Determine p ------------------------------------
    H00 = Hbra[0][0].ten.copy()
    if thermal:
        H00 = H00.transpose([0,1,2,3,6,4,5])
        H00mat = H00.reshape((np.prod(H00.shape[:5]),np.prod(H00.shape[5:])))
        _H00mat = lib.pinv(H00mat)
        _H00 = _H00mat.reshape(H00.shape[5:]+H00.shape[:5])
        _H00 = GEN_TEN(ten=_H00)
        _H00.merge_inds([4,5])
    else:
        H00 = H00.transpose([0,1,2,5,3,4])
        H00mat = H00.reshape((np.prod(H00.shape[:4]),np.prod(H00.shape[4:])))
        _H00mat = lib.pinv(H00mat)
        _H00 = _H00mat.reshape(H00.shape[4:]+H00.shape[:4])
        _H00 = GEN_TEN(ten=_H00)
    p = einsum('ldpru,RYldpu->RYr',Hbra_red[0][0],_H00)
    p.merge_inds([0,1])
    tmp = einsum('ldpru,rR->ldpRu',Hbra[0][0],p)

    # Determine q ------------------------------------
    H10 = Hbra[1][0].ten.copy()
    if thermal:
        H10 = H10.transpose([2,3,4,5,6,0,1])
        H10mat = H10.reshape((np.prod(H10.shape[:5]),np.prod(H10.shape[5:])))
        _H10mat = lib.pinv(H10mat)
        _H10 = _H10mat.reshape(H10.shape[5:]+H10.shape[:5])
        _H10 = GEN_TEN(ten=_H10)
        _H10.merge_inds([3,4])
    else:
        H10 = H10.transpose([2,3,4,5,0,1])
        H10mat = H10.reshape((np.prod(H10.shape[:4]),np.prod(H10.shape[4:])))
        _H10mat = lib.pinv(H10mat)
        _H10 = _H10mat.reshape(H10.shape[4:]+H10.shape[:4])
        _H10 = GEN_TEN(ten=_H10)
    q = einsum('ldpru,LYdpru->LYl',Hbra_red[1][0],_H10)
    q.merge_inds([0,1])
    tmp = einsum('ldpru,lL->Ldpru',Hbra[1][0],q)

    # Split singular values between the u/v and p/q tens --
    # Split u and v equally
    uv = einsum('ab,cb->ac',u,v)
    U,S,V = uv.svd(1,truncate_mbd=mbd,return_ent=False,return_wgt=False)
    u = einsum('ab,bc->ac',U,S.sqrt())
    v = einsum('ab,bc->ca',S.sqrt(),V)
    # Split p and q equally
    pq = einsum('ab,cb->ac',p,q)
    U,S,V = pq.svd(1,truncate_mbd=mbd,return_ent=False,return_wgt=False)
    p = einsum('ab,bc->ac',U,S.sqrt())
    q = einsum('ab,bc->ca',S.sqrt(),V)
    
    # Return Result!
    return u,v,p,q

def make_left_env(bra,ket,left,right,top,bot,truncate=False,chi=100):
    """
    Create a boundary mps for the left environment around the 
    p and q tensors

    Args:
        bra : list of two lists of length 2
            The peps tensors in the bra's unit cell with
            peps[0][0] -> bottom left tensor
            peps[0][1] -> top left tensor
            peps[1][0] -> bottom right tensor
            peps[1][1] -> top right tensor
        ket : list of two lists of length 2
            The peps tensors in the ket's unit cell with
            peps[0][0] -> bottom left tensor
            peps[0][1] -> top left tensor
            peps[1][0] -> bottom right tensor
            peps[1][1] -> top right tensor
        top : list of ndarrays
            The top environment tensors
        bot : list of ndarrays
            The bottom environment tensors
        left : list of ndarrays
            The left environment tensors
        right : list of ndarrays
            The right environment tensors

    Kwargs:
        truncate : bool
            Whether to truncate the environment tensors locally 
            when doing the ALS procedure
        chi : int
            If trunc_env == True, then this is the value that the 
            boundary MPS bond dimension will be reduced to

    Returns:
        Nl : MPS object
            The MPS object representing the left boundary mpo
            around the given unit cell
    """
    mpiprint(2,"\t\t\tMaking Left Environment")
    # Determine if it is a thermal state
    thermal = len(bra[0][0].legs[2]) == 2
    # Contract first layer --------------------------------------------------------------
    Nl = [None,None,None,None,None,None]
    # bottom bmpo site -------------------------------
    Nl[0] = einsum('ach,hns->ascn',bot[0],bot[1])
    Nl[0].merge_inds([2,3])
    # next bmpo site --------------------------------
    # TODO - Make sure signs are correct 
    # (will give error in symmetric case
    Db = ket[0][0].shape[ket[0][0].legs[1][0]]
    Zb = ket[0][0].qn_sectors[ket[0][0].legs[1][0]]
    I1 = eye(Db,Zb,is_symmetric=ket[0][0].is_symmetric,backend=ket[0][0].backend)
    if len(ket[0][0].legs[1]) > 1:
        for legind in range(1,len(ket[0][0].legs[1])):
            Dbi = ket[0][0].shape[ket[0][0].legs[1][legind]]
            Zbi = ket[0][0].qn_sectors[ket[0][0].legs[1][legind]]
            Ii = eye(Dbi,Zbi,is_symmetric=ket[0][0].is_symmetric,backend=ket[0][0].backend)
            I1 = einsum('ij,IJ->iIjJ',I1,Ii)
            I1.merge_inds([0,1])
            I1.merge_inds([1,2])
    Nl[1] = einsum('pfq,iI->pifqI',left[0],I1)
    Nl[1].merge_inds([0,1])
    Nl[1].merge_inds([2,3])
    # next bmpo site --------------------------------
    Nl[2] = einsum('dje,jouvp->douvep',left[1],ket[0][0])
    Nl[2].merge_inds([0,1])
    Nl[2].merge_inds([1,2])
    Nl[2].merge_inds([2,3])
    # next bmpo site --------------------------------
    # TODO - Make sure signs are correct 
    # (will give error in symmetric case
    Db = ket[0][1].shape[ket[0][1].legs[1][0]]
    Zb = ket[0][1].qn_sectors[ket[0][1].legs[1][0]]
    I1 = eye(Db,Zb,is_symmetric=ket[0][1].is_symmetric,backend=ket[0][1].backend)
    if len(ket[0][1].legs[1]) > 1:
        for legind in range(1,len(ket[0][1].legs[1])):
            Dbi = ket[0][1].shape[ket[0][1].legs[1][legind]]
            Zbi = ket[0][1].qn_sectors[ket[0][1].legs[1][legind]]
            Ii = eye(Dbi,Zbi,is_symmetric=ket[0][1].is_symmetric,backend=ket[0][1].backend)
            I1 = einsum('ij,IJ->iIjJ',I1,Ii)
            I1.merge_inds([0,1])
            I1.merge_inds([1,2])
    Nl[3] = einsum('ekf,pq->epkfq',left[2],I1)
    Nl[3].merge_inds([0,1])
    Nl[3].merge_inds([2,3])
    # next bmpo site --------------------------------
    Nl[4] = einsum('flg,lqxyr->fqxygr',left[3],ket[0][1])
    Nl[4].merge_inds([0,1])
    Nl[4].merge_inds([1,2])
    Nl[4].merge_inds([2,3])
    # last bmpo site --------------------------------
    Nl[5] = einsum('bgm,mrz->grzb',top[0],top[1])
    Nl[5].merge_inds([0,1])
    # Truncate bmpo --------------------------------
    Nl = MPS(Nl)
    if truncate:
        mpiprint(5,'Truncating Boundary MPS')
        if DEBUG:
            mpiprint(6,'Computing initial bmpo norm')
            norm0 = Nl.norm()
        Nl = Nl.apply_svd(chi)
        if DEBUG:
            mpiprint(6,'Computing resulting bmpo norm')
            norm1 = Nl.norm()
            mpiprint(0,'Norm Difference for chi={}: {}'.format(chi,abs(norm0-norm1)/abs(norm0)))
    # Contract second layer --------------------------------------------------------------
    Nl = Nl.tensors
    # bottom bmpo site -------------------------------
    Nl[0] = einsum('asn,sAH->aHnA',Nl[0],bot[2])
    Nl[0].merge_inds([2,3])
    # next bmpo site --------------------------------
    Nl[1] = einsum('nio,iAuIC->nAIouC',Nl[1],bra[0][0])
    Nl[1].merge_inds([0,1])
    Nl[1].merge_inds([2,3,4])
    # next bmpo site --------------------------------
    # Create identity
    Du = bra[0][0].shape[bra[0][0].legs[4][0]]
    Zu = bra[0][0].qn_sectors[bra[0][0].legs[4][0]]
    I1 = eye(Du,Zu,is_symmetric=bra[0][0].is_symmetric,backend=bra[0][0].backend)
    if len(bra[0][0].legs[4]) > 1:
        for legind in range(1,len(bra[0][0].legs[4])):
            Dbi = bra[0][0].shape[bra[0][0].legs[4][legind]]
            Zbi = bra[0][0].qn_sectors[bra[0][0].legs[4][legind]]
            Ii = eye(Dbi,Zbi,is_symmetric=bra[0][0].is_symmetric,backend=bra[0][0].backend)
            I1 = einsum('ij,IJ->iIjJ',I1,Ii)
            I1.merge_inds([0,1])
            I1.merge_inds([1,2])
    # Remove physical from right legs
    if thermal:
        if len(Nl[2].legs[1]) == 3:
            Nl[2].unmerge_ind(1)
            Nl[2].merge_inds([1,2]) # Merge back two physical inds
        elif len(Nl[2].legs[1]) == 4:
            Nl[2].unmerge_ind(1)
            Nl[2].merge_inds([1,2]) # Merge back two physical inds
            Nl[2].merge_inds([2,3]) # Merge back two right inds
        else:
            raise NotImplementedError()
    else:
        if len(Nl[2].legs[1]) == 2:
            Nl[2].unmerge_ind(1)
        elif len(Nl[2].legs[1]) == 3:
            Nl[2].unmerge_ind(1)
            Nl[2].merge_inds([2,3]) # Merge two right inds
        else:
            raise NotImplementedError()
    # Contract results
    Nl[2] = einsum('ouvp,CO->ouCvpO',Nl[2],I1)
    Nl[2].merge_inds([0,1,2])
    Nl[2].merge_inds([2,3])
    # next bmpo site --------------------------------
    Nl[3] = einsum('pkq,kOxKF->pOKqxF',Nl[3],bra[0][1])
    Nl[3].merge_inds([0,1])
    Nl[3].merge_inds([2,3,4])
    # next bmpo site --------------------------------
    # Create identity
    Du = bra[0][1].shape[bra[0][1].legs[4][0]]
    Zu = bra[0][1].qn_sectors[bra[0][1].legs[4][0]]
    I1 = eye(Du,Zu,is_symmetric=bra[0][1].is_symmetric,backend=bra[0][1].backend)
    if len(bra[0][1].legs[4]) > 1:
        for legind in range(1,len(bra[0][1].legs[4])):
            Dbi = bra[0][1].shape[bra[0][1].legs[4][legind]]
            Zbi = bra[0][1].qn_sectors[bra[0][1].legs[4][legind]]
            Ii = eye(Dbi,Zbi,is_symmetric=bra[0][1].is_symmetric,backend=bra[0][1].backend)
            I1 = einsum('ij,IJ->iIjJ',I1,Ii)
            I1.merge_inds([0,1])
            I1.merge_inds([0,1])
    # Remove physical from right legs
    if thermal:
        if len(Nl[4].legs[1]) == 3:
            Nl[4].unmerge_ind(1)
            Nl[4].merge_inds([1,2]) # Merge back two physical inds
        elif len(Nl[4].legs[1]) == 4:
            Nl[4].unmerge_ind(1)
            Nl[4].merge_inds([1,2]) # Merge back two physical inds
            Nl[4].merge_inds([2,3]) # Merge back two right inds
        else:
            raise NotImplementedError()
    else:
        if len(Nl[4].legs[1]) == 2:
            Nl[4].unmerge_ind(1)
        elif len(Nl[4].legs[1]) == 3:
            Nl[4].unmerge_ind(1)
            Nl[4].merge_inds([2,3]) # Merge two right inds
        else:
            raise NotImplementedError()
    # Contract results
    Nl[4] = einsum('qxyr,FG->qxFyrG',Nl[4],I1)
    Nl[4].merge_inds([0,1,2])
    Nl[4].merge_inds([2,3])
    # last bmpo site --------------------------------
    Nl[5] = einsum('rzb,zGM->rGMb',Nl[5],top[2])
    Nl[5].merge_inds([0,1])
    # Truncate bmpo --------------------------------
    Nl = MPS(Nl)
    if truncate:
        mpiprint(5,'Truncating Boundary MPS')
        if DEBUG:
            mpiprint(6,'Computing initial bmpo norm')
            norm0 = Nl.norm()
        Nl = Nl.apply_svd(chi)
        if DEBUG:
            mpiprint(6,'Computing resulting bmpo norm')
            norm1 = Nl.norm()
            mpiprint(0,'Norm Difference for chi={}: {}'.format(chi,abs(norm0-norm1)/abs(norm0)))
    # Return result
    return Nl

def make_right_env(bra,ket,left,right,top,bot,truncate=False,chi=100):
    """
    Create a boundary mps for the right environment around the 
    p and q tensors

    Args:
        bra : list of two lists of length 2
            The peps tensors in the bra's unit cell with
            peps[0][0] -> bottom left tensor
            peps[0][1] -> top left tensor
            peps[1][0] -> bottom right tensor
            peps[1][1] -> top right tensor
        ket : list of two lists of length 2
            The peps tensors in the ket's unit cell with
            peps[0][0] -> bottom left tensor
            peps[0][1] -> top left tensor
            peps[1][0] -> bottom right tensor
            peps[1][1] -> top right tensor
        top : list of ndarrays
            The top environment tensors
        bot : list of ndarrays
            The bottom environment tensors
        left : list of ndarrays
            The left environment tensors
        right : list of ndarrays
            The right environment tensors

    Kwargs:
        truncate : bool
            Whether to truncate the environment tensors locally 
            when doing the ALS procedure
        chi : int
            If trunc_env == True, then this is the value that the 
            boundary MPS bond dimension will be reduced to

    Returns:
        Nr : MPS object
            The MPS object representing the left boundary mpo
            around the given unit cell
    """
    mpiprint(2,"\t\t\tMaking Right Environment")
    # Determine if it is a thermal state
    thermal = len(bra[0][0].legs[2]) == 2
    # Contract first layer --------------------------------------------------------------
    Nr = [None,None,None,None,None,None]
    # bottom bmpo site -------------------------------
    Nr[0] = einsum('hca,qnh->aqnc',bot[5],bot[4])
    Nr[0].merge_inds([2,3])
    # next bmpo site --------------------------------
    Nr[1] = einsum('cid,rnPio->ncrPod',right[0],bra[1][0])
    Nr[1].merge_inds([0,1])
    Nr[1].merge_inds([2,3,4])
    # next bmpo site --------------------------------
    # TODO - Make sure signs are correct 
    # (will give error in symmetric case
    Dr = bra[1][0].shape[bra[1][0].legs[4][0]]
    Zr = bra[1][0].qn_sectors[bra[1][0].legs[4][0]]
    I1 = eye(Dr,Zr,is_symmetric=bra[1][0].is_symmetric,backend=bra[1][0].backend)
    if len(bra[1][0].legs[4]) > 1:
        for legind in range(1,len(bra[1][0].legs[4])):
            Dri = bra[1][0].shape[bra[1][0].legs[4][legind]]
            Zri = bra[1][0].qn_sectors[bra[1][0].legs[4][legind]]
            Ii = eye(Dri,Zri,is_symmetric=bra[1][0].is_symmetric,backend=bra[1][0].backend)
            I1 = einsum('ij,IJ->iIjJ',I1,Ii)
            I1.merge_inds([0,1])
            I1.merge_inds([1,2])
    Dp = bra[1][0].shape[bra[1][0].legs[2][0]]
    Zp = bra[1][0].qn_sectors[bra[1][0].legs[2][0]]
    I2 = eye(Dp,Zp,is_symmetric=bra[1][0].is_symmetric,backend=bra[1][0].backend)
    if len(bra[1][0].legs[2]) > 1:
        for legind in range(1,len(bra[1][0].legs[2])):
            Dpi = bra[1][0].shape[bra[1][0].legs[2][legind]]
            Zpi = bra[1][0].qn_sectors[bra[1][0].legs[2][legind]]
            Ii = eye(Dpi,Zpi,is_symmetric=bra[1][0].is_symmetric,backend=bra[1][0].backend)
            I2 = einsum('ij,IJ->iIjJ',I2,Ii)
            I2.merge_inds([0,1])
            I2.merge_inds([1,2])
    Nr[2] = einsum('dje,Oo->djeOo',right[1],I1)
    Nr[2] = einsum('djeOo,Pp->POdpjoe',Nr[2],I2)
    Nr[2].merge_inds([0,1,2])
    Nr[2].merge_inds([1,2])
    Nr[2].merge_inds([2,3])
    # next bmpo site --------------------------------
    Nr[3] = einsum('ekf,soQkp->oesQpf',right[2],bra[1][1])
    Nr[3].merge_inds([0,1])
    Nr[3].merge_inds([2,3,4])
    # next bmpo site --------------------------------
    Dr = bra[1][1].shape[bra[1][1].legs[4][0]]
    Zr = bra[1][1].qn_sectors[bra[1][1].legs[4][0]]
    I1 = eye(Dr,Zr,is_symmetric=bra[1][1].is_symmetric,backend=bra[1][1].backend)
    if len(bra[1][1].legs[4]) > 1:
        for legind in range(1,len(bra[1][1].legs[4])):
            Dri = bra[1][1].shape[bra[1][1].legs[4][legind]]
            Zri = bra[1][1].qn_sectors[bra[1][1].legs[4][legind]]
            Ii = eye(Dri,Zri,is_symmetric=bra[1][1].is_symmetric,backend=bra[1][1].backend)
            I1 = einsum('ij,IJ->iIjJ',I1,Ii)
            I1.merge_inds([0,1])
            I1.merge_inds([1,2])
    Dp = bra[1][1].shape[bra[1][1].legs[2][0]]
    Zp = bra[1][1].qn_sectors[bra[1][1].legs[2][0]]
    I2 = eye(Dp,Zp,is_symmetric=bra[1][1].is_symmetric,backend=bra[1][1].backend)
    if len(bra[1][1].legs[2]) > 1:
        for legind in range(1,len(bra[1][1].legs[2])):
            Dpi = bra[1][1].shape[bra[1][1].legs[2][legind]]
            Zpi = bra[1][1].qn_sectors[bra[1][1].legs[2][legind]]
            Ii = eye(Dpi,Zpi,is_symmetric=bra[1][1].is_symmetric,backend=bra[1][1].backend)
            I2 = einsum('ij,IJ->iIjJ',I2,Ii)
            I2.merge_inds([0,1])
            I2.merge_inds([1,2])
    Nr[4] = einsum('flg,Pp->flgPp',right[3],I1)
    Nr[4] = einsum('flgPp,Qq->QPfqlpg',Nr[4],I2)
    Nr[4].merge_inds([0,1,2])
    Nr[4].merge_inds([1,2])
    Nr[4].merge_inds([2,3])
    # last bmpo site --------------------------------
    Nr[5] = einsum('mgb,tpm->pgtb',top[5],top[4])
    Nr[5].merge_inds([0,1])
    # Truncate bmpo --------------------------------
    Nr = MPS(Nr)
    if truncate:
        mpiprint(5,'Truncating Boundary MPS')
        if DEBUG:
            mpiprint(6,'Computing initial bmpo norm')
            norm0 = Nr.norm()
        Nr = Nr.apply_svd(chi)
        if DEBUG:
            mpiprint(6,'Computing resulting bmpo norm')
            norm1 = Nr.norm()
            mpiprint(0,'Norm Difference for chi={}: {}'.format(chi,abs(norm0-norm1)/abs(norm0)))
    # Contract second layer --------------------------------------------------------------
    Nr = Nr.tensors
    # bottom bmpo site -------------------------------
    Nr[0] = einsum('aqn,xuq->axun',Nr[0],bot[3])
    Nr[0].merge_inds([2,3])
    # next bmpo site --------------------------------
    Du = ket[1][0].shape[ket[1][0].legs[1][0]]
    Zu = ket[1][0].qn_sectors[ket[1][0].legs[1][0]]
    I1 = eye(Du,Zu,is_symmetric=ket[1][0].is_symmetric,backend=ket[1][0].backend)
    if len(ket[1][0].legs[1]) > 1:
        for legind in range(1,len(ket[1][0].legs[1])):
            Dui = ket[1][0].shape[ket[1][0].legs[1][legind]]
            Zui = ket[1][0].qn_sectors[ket[1][0].legs[1][legind]]
            Ii = eye(Dui,Zui,is_symmetric=ket[1][0].is_symmetric,backend=ket[1][0].backend)
            I1 = einsum('ij,IJ->iIjJ',I1,Ii)
            I1.merge_inds([0,1])
            I1.merge_inds([1,2])
    Nr[1] = einsum('nro,Uu->Unruo',Nr[1],I1)
    Nr[1].merge_inds([0,1])
    Nr[1].merge_inds([2,3])
    # next bmpo site --------------------------------
    Nr[2].unmerge_ind(1)
    if thermal:
        Nr[2].merge_inds([1,2])
    Nr[2] = einsum('opjO,yupjv->uoyvO',Nr[2],ket[1][0])
    Nr[2].merge_inds([0,1])
    Nr[2].merge_inds([2,3])
    # next bmpo site --------------------------------
    Du = ket[1][1].shape[ket[1][1].legs[1][0]]
    Zu = ket[1][1].qn_sectors[ket[1][1].legs[1][0]]
    I1 = eye(Du,Zu,is_symmetric=ket[1][1].is_symmetric,backend=ket[1][1].backend)
    if len(ket[1][1].legs[1]) > 1:
        for legind in range(1,len(ket[1][1].legs[1])):
            Dui = ket[1][1].shape[ket[1][1].legs[1][legind]]
            Zui = ket[1][1].qn_sectors[ket[1][1].legs[1][legind]]
            Ii = eye(Dui,Zui,is_symmetric=ket[1][1].is_symmetric,backend=ket[1][1].backend)
            I1 = einsum('ij,IJ->iIjJ',I1,Ii)
            I1.merge_inds([0,1])
            I1.merge_inds([1,2])
    Nr[3] = einsum('osp,Vv->Vosvp',Nr[3],I1)
    Nr[3].merge_inds([0,1])
    Nr[3].merge_inds([2,3])
    # next bmpo site --------------------------------
    Nr[4].unmerge_ind(1)
    if thermal:
        Nr[4].merge_inds([1,2])
    Nr[4] = einsum('pqlP,zvqlw->vpzwP',Nr[4],ket[1][1])
    Nr[4].merge_inds([0,1])
    Nr[4].merge_inds([2,3])
    # last bmpo site --------------------------------
    Nr[5] = einsum('ptb,awt->wpab',Nr[5],top[3])
    Nr[5].merge_inds([0,1])
    # Truncate bmpo --------------------------------
    Nr = MPS(Nr)
    if truncate:
        mpiprint(5,'Truncating Boundary MPS')
        if DEBUG:
            mpiprint(6,'Computing initial bmpo norm')
            norm0 = Nr.norm()
        Nr = Nr.apply_svd(chi)
        if DEBUG:
            mpiprint(6,'Computing resulting bmpo norm')
            norm1 = Nr.norm()
            mpiprint(0,'Norm Difference for chi={}: {}'.format(chi,abs(norm0-norm1)/abs(norm0)))
    # Return result
    return Nr


def calc_NK_lb_pq(peps,eH,top,bot,left,right,u,v,truncate=False,chi=100):
    """
    Calculates the two environment tensors (here N and K) which 
    are used in the alternating least squares procedure for determining
    p and q

    Args:
        peps : list of two lists of length 2
            The peps tensors in the unit cell with
            peps[0][0] -> bottom left tensor
            peps[0][1] -> top left tensor
            peps[1][0] -> bottom right tensor
            peps[1][1] -> top right tensor
        eH : list of three tensors
            The exponentiated MPO for the three sites 
        top : list of ndarrays
            The top environment tensors
        bot : list of ndarrays
            The bottom environment tensors
        left : list of ndarrays
            The left environment tensors
        right : list of ndarrays
            The right environment tensors
        u : ndarray
            The reducing tensor for the bottom leg of the top left tensor
        v : ndarray
            The reducing tensor for the top leg of the bottom left tensor

    Kwargs:
        truncate : bool
            Whether to truncate the environment tensors locally 
            when doing the ALS procedure
        chi : int
            If trunc_env == True, then this is the value that the 
            boundary MPS bond dimension will be reduced to
        make_pos_herm : bool
            If True (default) then the norm will be approximated by
            as hermitian and positive

    returns:
        N: Tensor
            The full environment tensor around u and v (or p and q)
            Currently stored with legs in order
            bra-left,bra-right,ket-left,ket-right
        K: Tensor
            The environment tensor around only the ket's instance
            of u and v (or p and q)
            Currently stored with legs in order
            bra-left,bra-right,ket-left,ket-right
            where the ket is the full evolved state
            and the bra is the truncated evolved state
    """
    mpiprint(2,"\t\tEvolving State")
    # Check if dealing with a thermal state
    thermal = len(peps[0][0].legs[2]) == 2
    # Contract state with time evolution operator
    Hbra = [[None,None],[None,None]]
    if thermal:
        peps[0][1].unmerge_ind(2)
        Hbra[0][1] = einsum('ldparu,pPx->ldxParu',peps[0][1],eH[0])
        Hbra[0][1].merge_inds([3,4])
        peps[0][1].merge_inds([2,3])
        peps[0][0].unmerge_ind(2)
        Hbra[0][0] = einsum('ldparu,xpPy->ldParyux',peps[0][0],eH[1])
        Hbra[0][0].merge_inds([2,3])
        peps[0][0].merge_inds([2,3])
        peps[1][0].unmerge_ind(2)
        Hbra[1][0] = einsum('ldparu,ypP->lydParu',peps[1][0],eH[2])
        Hbra[1][0].merge_inds([3,4])
        peps[1][0].merge_inds([2,3])
        Hbra[1][1] = peps[1][1].copy()
    else:
        Hbra[0][1] = einsum('ldpru,pPx->ldxPru',peps[0][1],eH[0]) # Top left site
        Hbra[0][0] = einsum('ldpru,xpPy->ldPryux',peps[0][0],eH[1]) # Bottom left site
        Hbra[1][0] = einsum('ldpru,ypP->lydPru',peps[1][0],eH[2]) # Bottom right site
        Hbra[1][1] = peps[1][1].copy()
    # Merge indices as needed
    Hbra[0][1].merge_inds([1,2])
    Hbra[0][0].merge_inds([3,4])
    Hbra[0][0].merge_inds([4,5])
    Hbra[1][0].merge_inds([0,1])
    # Contract state with reducing tensors on unused bonds
    Hbra_red = [[None,None],[None,None]]
    Hbra_red[0][1] = einsum('ldpru,dD->lDpru',Hbra[0][1],u)
    Hbra_red[0][0] = einsum('ldpru,uU->ldprU',Hbra[0][0],v)
    Hbra_red[1][0] = Hbra[1][0].copy()
    Hbra_red[1][1] = Hbra[1][1].copy()
    # Replace top and bottoms with ones if needed ----
    if bot is None:
        bot = fill_empty_bot(peps,left,right)
    if top is None:
        top = fill_empty_top(peps,left,right)
    # Contract environment around p and q tensors
    mpiprint(2,"\t\tCalculating N")
    if False:
        # Slower contraction method
        # Contract right half
        Nr = einsum('hca,cid->ahid',bot[5],right[0]).remove_empty_ind(0)
        Nr = einsum('hid,dje->hije',Nr,right[1])
        Nr = einsum('hije,ekf->hijkf',Nr,right[2])
        Nr = einsum('hijkf,flg->hijklg',Nr,right[3])
        Nr = einsum('hijklg,mgb->bhijklm',Nr,top[5]).remove_empty_ind(0)
        Nr = einsum('hijklm,qnh->qnijklm',Nr,bot[4])
        Nr = einsum('qnijklm,rnPio->qrPojklm',Nr,Hbra_red[1][0])
        Nr = einsum('qrPojklm,soQkp->qrPjsQplm',Nr,Hbra_red[1][1])
        Nr = einsum('qrPjsQplm,tpm->qrPjsQlt',Nr,top[4])
        Nr = einsum('qrPjsQlt,xuq->xurPjsQlt',Nr,bot[3])
        Nr = einsum('xurPjsQlt,yuPjv->xryvsQlt',Nr,Hbra_red[1][0])
        Nr = einsum('xryvsQlt,zvQlw->xryszwt',Nr,Hbra_red[1][1])
        Nr = einsum('xryszwt,awt->xrysza',Nr,top[3])
        # Contract left half
        Nl = einsum('upl,pfq->ulfq',bot[0],left[0]).remove_empty_ind(0)
        Nl = einsum('lfq,qmr->lfmr',Nl,left[1])
        Nl = einsum('lfmr,rgs->lfmgs',Nl,left[2])
        Nl = einsum('lfmgs,snt->lfmgnt',Nl,left[3])
        Nl = einsum('lfmgnt,vto->vlfmgno',Nl,top[0]).remove_empty_ind(0)
        Nl = einsum('lfmgno,lie->eifmgno',Nl,bot[1])
        Nl = einsum('eifmgno,miPyj->efPyjgno',Nl,Hbra_red[0][0])
        Nl = einsum('efPyjgno,njQzk->efPygQzko',Nl,Hbra_red[0][1])
        Nl = einsum('efPygQzko,okh->efPygQzh',Nl,top[1])
        Nl = einsum('efPygQzh,ebx->xbfPygQzh',Nl,bot[2])
        Nl = einsum('xbfPygQzh,fbPrc->xrcygQzh',Nl,Hbra_red[0][0])
        Nl = einsum('xrcygQzh,gcQsd->xrysdzh',Nl,Hbra_red[0][1])
        Nl = einsum('xrysdzh,hda->xrysza',Nl,top[2])
        # Combine left and right halves
        N = einsum('xrysza,xRYsza->rRyY',Nl,Nr)
    else:
        # Faster Contraction Method
        # Contract right half
        Nr = make_right_env(Hbra_red,
                            Hbra_red,
                            left,
                            right,
                            top,
                            bot,
                            truncate=truncate,
                            chi=chi)
        # Contract left half
        Nl = make_left_env(Hbra_red,
                           Hbra_red,
                           left,
                           right,
                           top,
                           bot,
                           truncate=truncate,
                           chi=chi)
        # Combine left and right halves
        N = einsum('apb,ApB->aAbB',Nl[5],Nr[5]).remove_empty_ind(3).remove_empty_ind(2)
        N = einsum('apb,bB->apB',Nl[4],N)
        N = einsum('ApB,apB->aA',Nr[4],N)
        N = einsum('apb,bB->apB',Nl[3],N)
        N = einsum('ApB,apB->aA',Nr[3],N)
        N = einsum('axb,bB->axB',Nl[2],N)
        N = einsum('AXB,axB->aAxX',Nr[2],N)
        N = einsum('ayb,bBxX->ayBxX',Nl[1],N)
        N = einsum('AYB,ayBxX->aAxXyY',Nr[1],N)
        N = einsum('apb,bBxXyY->apBxXyY',Nl[0],N)
        N = einsum('ApB,apBxXyY->aAyYxX',Nr[0],N).remove_empty_ind(0).remove_empty_ind(0)
    # Contract environment around p and q tensors
    mpiprint(2,"\t\tCalculating K")
    if False:
        # Contract right half
        Kr = einsum('hca,cid->ahid',bot[5],right[0]).remove_empty_ind(0)
        Kr = einsum('hid,dje->hije',Kr,right[1])
        Kr = einsum('hije,ekf->hijkf',Kr,right[2])
        Kr = einsum('hijkf,flg->hijklg',Kr,right[3])
        Kr = einsum('hijklg,mgb->bhijklm',Kr,top[5]).remove_empty_ind(0)
        Kr = einsum('hijklm,qnh->qnijklm',Kr,bot[4])
        Kr = einsum('qnijklm,rnPio->qrPojklm',Kr,Hbra_red[1][0])
        Kr = einsum('qrPojklm,soQkp->qrPjsQplm',Kr,Hbra_red[1][1])
        Kr = einsum('qrPjsQplm,tpm->qrPjsQlt',Kr,top[4])
        Kr = einsum('qrPjsQlt,xuq->xurPjsQlt',Kr,bot[3])
        Kr = einsum('xurPjsQlt,yuPjv->xryvsQlt',Kr,Hbra[1][0])
        Kr = einsum('xryvsQlt,zvQlw->xryszwt',Kr,Hbra[1][1])
        Kr = einsum('xryszwt,awt->xrysza',Kr,top[3])
        # Contract left half
        Kl = einsum('upl,pfq->ulfq',bot[0],left[0]).remove_empty_ind(0)
        Kl = einsum('lfq,qmr->lfmr',Kl,left[1])
        Kl = einsum('lfmr,rgs->lfmgs',Kl,left[2])
        Kl = einsum('lfmgs,snt->lfmgnt',Kl,left[3])
        Kl = einsum('lfmgnt,vto->vlfmgno',Kl,top[0]).remove_empty_ind(0)
        Kl = einsum('lfmgno,lie->eifmgno',Kl,bot[1])
        Kl = einsum('eifmgno,miPyj->efPyjgno',Kl,Hbra[0][0])
        Kl = einsum('efPyjgno,njQzk->efPygQzko',Kl,Hbra[0][1])
        Kl = einsum('efPygQzko,okh->efPygQzh',Kl,top[1])
        Kl = einsum('efPygQzh,ebx->xbfPygQzh',Kl,bot[2])
        Kl = einsum('xbfPygQzh,fbPrc->xrcygQzh',Kl,Hbra_red[0][0])
        Kl = einsum('xrcygQzh,gcQsd->xrysdzh',Kl,Hbra_red[0][1])
        Kl = einsum('xrysdzh,hda->xrysza',Kl,top[2])
        K = einsum('xrysza,xRysza->rR',Kl,Kr)
    else:
        # Contract right half
        Kr = make_right_env(Hbra_red,
                            Hbra,
                            left,
                            right,
                            top,
                            bot,
                            truncate=truncate,
                            chi=chi)
        # Contract left half
        Kl = make_left_env(Hbra_red,
                           Hbra,
                           left,
                           right,
                           top,
                           bot,
                           truncate=truncate,
                           chi=chi)
        # Combine left and right halves
        K = einsum('apb,ApB->aAbB',Kl[5],Kr[5]).remove_empty_ind(3).remove_empty_ind(2)
        K = einsum('apb,bB->apB',Kl[4],K)
        K = einsum('ApB,apB->aA',Kr[4],K)
        K = einsum('apb,bB->apB',Kl[3],K)
        K = einsum('ApB,apB->aA',Kr[3],K)
        K = einsum('apb,bB->apB',Kl[2],K)
        K = einsum('ApB,apB->aA',Kr[2],K)
        K = einsum('ayb,bB->ayB',Kl[1],K)
        K = einsum('AYB,ayB->aAyY',Kr[1],K)
        K = einsum('apb,bByY->apByY',Kl[0],K)
        K = einsum('ApB,apByY->aAyY',Kr[0],K).remove_empty_ind(0).remove_empty_ind(0)

    # Return result
    return N,K

def calc_NK_lb_uv(peps,eH,top,bot,left,right,p,q,truncate=False,chi=100):
    """
    Calculates the two environment tensors (here N and K) which 
    are used in the alternating least squares procedure for determining
    u and v

    Args:
        peps : list of two lists of length 2
            The peps tensors in the unit cell with
            peps[0][0] -> bottom left tensor
            peps[0][1] -> top left tensor
            peps[1][0] -> bottom right tensor
            peps[1][1] -> top right tensor
        eH : list of three tensors
            The exponentiated MPO for the three sites 
        top : list of ndarrays
            The top environment tensors
        bot : list of ndarrays
            The bottom environment tensors
        left : list of ndarrays
            The left environment tensors
        right : list of ndarrays
            The right environment tensors
        p : ndarray
            The reducing tensor for the right leg of the bottom left tensor
        q : ndarray 
            The reducing tensor for the left leg of the bottom right tensor

    Kwargs:
        truncate : bool
            Whether to truncate the environment tensors locally 
            when doing the ALS procedure
        chi : int
            If trunc_env == True, then this is the value that the 
            boundary MPS bond dimension will be reduced to
        make_pos_herm : bool
            If True (default) then the norm will be approximated by
            as hermitian and positive

    returns:
        N: Tensor
            The full environment tensor around u and v (or p and q)
            Currently stored with legs in order
            bra-left,bra-right,ket-left,ket-right
        K: Tensor
            The environment tensor around only the ket's instance
            of u and v (or p and q)
            Currently stored with legs in order
            bra-left,bra-right,ket-left,ket-right
            where the ket is the full evolved state
            and the bra is the truncated evolved state

    """
    mpiprint(2,"\t\tEvolving State")
    # Check if dealing with a thermal state
    thermal = len(peps[0][0].legs[2]) == 2
    # Contract state with time evolution operator
    Hbra = [[None,None],[None,None]]
    if thermal:
        peps[0][1].unmerge_ind(2)
        Hbra[0][1] = einsum('ldparu,pPx->ldxParu',peps[0][1],eH[0])
        Hbra[0][1].merge_inds([3,4])
        peps[0][1].merge_inds([2,3])
        peps[0][0].unmerge_ind(2)
        Hbra[0][0] = einsum('ldparu,xpPy->ldParyux',peps[0][0],eH[1])
        Hbra[0][0].merge_inds([2,3])
        peps[0][0].merge_inds([2,3])
        peps[1][0].unmerge_ind(2)
        Hbra[1][0] = einsum('ldparu,ypP->lydParu',peps[1][0],eH[2])
        Hbra[1][0].merge_inds([3,4])
        peps[1][0].merge_inds([2,3])
        Hbra[1][1] = peps[1][1].copy()
    else:
        Hbra[0][1] = einsum('ldpru,pPx->ldxPru',peps[0][1],eH[0]) # Top left site
        Hbra[0][0] = einsum('ldpru,xpPy->ldPryux',peps[0][0],eH[1]) # Bottom left site
        Hbra[1][0] = einsum('ldpru,ypP->lydPru',peps[1][0],eH[2]) # Bottom right site
        Hbra[1][1] = peps[1][1].copy()
    # Merge indices as needed
    Hbra[0][1].merge_inds([1,2])
    Hbra[0][0].merge_inds([3,4])
    Hbra[0][0].merge_inds([4,5])
    Hbra[1][0].merge_inds([0,1])
    # Contract state with reducing tensors on unused bonds
    Hbra_red = [[None,None],[None,None]]
    Hbra_red[0][0] = einsum('ldpru,rR->ldpRu',Hbra[0][0],p)
    Hbra_red[1][0] = einsum('ldpru,lL->Ldpru',Hbra[1][0],q)
    Hbra_red[0][1] = Hbra[0][1].copy()
    Hbra_red[1][1] = Hbra[1][1].copy()
    # Calculate M ------------------------------------
    mpiprint(2,"\t\tCalculating N")
    mpiprint(2,"\t\t\tBot Environment")
    Nbot = update_bot_env2(0,
                           Hbra_red,
                           Hbra_red,
                           left[0],
                           left[1],
                           right[0],
                           right[1],
                           bot,
                           truncate=truncate,
                           chi=chi)
    mpiprint(2,"\t\t\tTop Environment")
    Ntop = update_top_env2(1,
                           Hbra_red,
                           Hbra_red,
                           left[2],
                           left[3],
                           right[2],
                           right[3],
                           top,
                           truncate=truncate,
                           chi=chi)
    # Now contract the two bmps around u
    mpiprint(2,"\t\t\tContracting MPS")
    N = einsum('apb,ApB->aAbB',Nbot[5],Ntop[5]).remove_empty_ind(3).remove_empty_ind(2)
    N = einsum('apb,bB->apB',Nbot[4],N)
    N = einsum('ApB,apB->aA',Ntop[4],N)
    N = einsum('apb,bB->apB',Nbot[3],N)
    N = einsum('ApB,apB->aA',Ntop[3],N)
    N = einsum('axb,bB->axB',Nbot[2],N)
    N = einsum('AXB,axB->aAxX',Ntop[2],N)
    N = einsum('ayb,bBxX->ayBxX',Nbot[1],N)
    N = einsum('AYB,ayBxX->aAxXyY',Ntop[1],N)
    N = einsum('apb,bBxXyY->apBxXyY',Nbot[0],N)
    N = einsum('ApB,apBxXyY->aAxXyY',Ntop[0],N).remove_empty_ind(0).remove_empty_ind(0)
    # Calculate K ------------------------------------
    mpiprint(2,"\t\t\tBot Environment")
    Kbot = update_bot_env2(0,
                           Hbra_red,
                           Hbra,
                           left[0],
                           left[1],
                           right[0],
                           right[1],
                           bot,
                           truncate=truncate,
                           chi=chi)
    mpiprint(2,"\t\t\tTop Environment")
    Ktop = update_top_env2(1,
                           Hbra_red,
                           Hbra,
                           left[2],
                           left[3],
                           right[2],
                           right[3],
                           top,
                           truncate=truncate,
                           chi=chi)
    # Now contract the two bmps around u
    mpiprint(2,"\t\t\tContracting MPS")
    K = einsum('apb,ApB->aAbB',Kbot[5],Ktop[5]).remove_empty_ind(3).remove_empty_ind(2)
    K = einsum('apb,bB->apB',Kbot[4],K)
    K = einsum('ApB,apB->aA',Ktop[4],K)
    K = einsum('apb,bB->apB',Kbot[3],K)
    K = einsum('ApB,apB->aA',Ktop[3],K)
    K = einsum('axb,bB->axB',Kbot[2],K)
    K = einsum('AXB,axB->aAXx',Ktop[2],K)
    K = einsum('apb,bBXx->apBXx',Kbot[1],K)
    K = einsum('ApB,apBXx->aAXx',Ktop[1],K)
    K = einsum('apb,bBXx->apBXx',Kbot[0],K)
    K = einsum('ApB,apBXx->aAxX',Ktop[0],K).remove_empty_ind(0).remove_empty_ind(0)
    
    # Return results
    return N,K

def calc_NK(peps,eH,top,bot,left,right,u,v,p,q,lb=True,pq=True,truncate=False,chi=100):
    """
    Calculates the two environment tensors (here N and K) which 
    are used in the alternating least squares procedure

    Args:
        peps : list of two lists of length 2
            The peps tensors in the unit cell with
            peps[0][0] -> bottom left tensor
            peps[0][1] -> top left tensor
            peps[1][0] -> bottom right tensor
            peps[1][1] -> top right tensor
        eH : list of three tensors
            The exponentiated MPO for the three sites 
        top : list of ndarrays
            The top environment tensors
        bot : list of ndarrays
            The bottom environment tensors
        left : list of ndarrays
            The left environment tensors
        right : list of ndarrays
            The right environment tensors
        u : ndarray
            The reducing tensor for the bottom leg of the top left tensor
        v : ndarray
            The reducing tensor for the top leg of the bottom left tensor
        p : ndarray
            The reducing tensor for the right leg of the bottom left tensor
        q : ndarray 
            The reducing tensor for the left leg of the bottom right tensor

    Kwargs:
        lb : bool
            Whether the MPO is acting on the left bottom edges of the 
            PEPS 2x2 unit
        pq : bool
            Whether the environments should be contracted around
            p and q (True) or around u and v (False)
        truncate : bool
            Whether to truncate the environment tensors locally 
            when doing the ALS procedure
        chi : int
            If trunc_env == True, then this is the value that the 
            boundary MPS bond dimension will be reduced to

    Returns:
        N : ndarray
            The full environment tensor around u and v (or p and q)
        K : ndarray
            The environment tensor around only the ket's instance
            of u and v (or p and q)
    """
    if lb:
        if pq:
            return calc_NK_lb_pq(peps,eH,top,bot,left,right,u,v,truncate=truncate,chi=chi)
        else:
            return calc_NK_lb_uv(peps,eH,top,bot,left,right,p,q,truncate=truncate,chi=chi)
    else:
        raise NotImplementedError()

def calc_cost_uv_lb(N,K,u,v):
    """
    """
    # Contract N with the u and v tensors
    tt = einsum('xXyY,yz->xXzY',N,v)
    tt = einsum('xXyY,xz->zXyY',tt,v)
    tt = einsum('xXyY,Xx->yY',tt,u)
    tt = einsum('yY,Yy->',tt,u)
    # Contract K with the u and v tensors
    tp = einsum('xX,xz->zX',K,v)
    tp = einsum('xX,Xx->',tp,u)
    return 1.+tt-2.*tp

def calc_cost_pq_lb(N,K,p,q):
    """
    """
    # Contract N with the p and q tensors
    tt = einsum('rRyY,YZ->rRyZ',N,q)
    tt = einsum('rRyY,RZ->rZyY',tt,q)
    tt = einsum('rRyY,yY->rR',tt,p)
    tt = einsum('rR,rR->',tt,p)
    # Contract K with the p and q tensors
    tp = einsum('rR,RS->rS',K,q)
    tp = einsum('rR,rR->',tp,p)
    return 1.+tt-2.*tp

def optimize_u_lb(N,K,u,v):
    """
    Use least squares procedure to determine the u tensor
    """
    # Calculate R (similar to https://arxiv.org/pdf/1503.05345.pdf)
    # by contracting v into N
    R = einsum('xXyY,yz->xXzY',N,v)
    R = einsum('xXyY,xz->zXyY',R,v)
    # Calculate S (again similar to paper above) 
    # by contracting v into K
    S = einsum('xX,xz->zX',K,v)
    # Convert R into a matrix
    Rmat = R.ten.reshape((np.prod(R.shape[:3]),-1))
    # Convert S into a vector
    Svec = S.ten.reshape((-1))
    # Solve R*u=S
    uvec,_,_,_ = np.linalg.lstsq(Rmat,Svec,rcond=None)
    # Put u back into a symtensor
    u.ten = uvec.reshape(S.shape).transpose([1,2,0])
    # Evaluate the results
    cost = calc_cost_uv_lb(N,K,u,v)
    # Return result
    return u,cost

def optimize_v_lb(N,K,u,v):
    """
    Use least squares procedure to determine the v tensor
    """
    # Calculate R (similar to https://arxiv.org/pdf/1503.05345.pdf)
    # by contracting u into N
    R = einsum('xXyY,YZ->xXyZ',N,u)
    R = einsum('xXyY,XZ->xZyY',R,u)
    # Calculate S (again similar to paper above) 
    # by contracting u into K
    S = einsum('xX,XZ->xZ',K,u)
    # Convert R into a matrix
    Rmat = R.ten.reshape((np.prod(R.shape[:3]),-1))
    # Convert S into a vector
    Svec = S.ten.reshape((-1))
    # Solve R*v=S
    vvec,_,_,_ = np.linalg.lstsq(Rmat,Svec,rcond=None)
    # Put v back into a symtensor
    v.ten = vvec.reshape(S.shape)
    # Evaluate the results
    cost = calc_cost_uv_lb(N,K,u,v)
    # Return result
    return v,cost

def optimize_p_lb(N,K,p,q):
    """
    Use least squares procedure to determine the p tensor
    """
    # Calculate R (similar to to https://arxiv.org/pdf/1503.05345.pdf)
    # by contracting q into N
    R = einsum('rRyY,YZ->rRyZ',N,q)
    R = einsum('rRyY,RZ->rZyY',R,q)
    # Calculate S (again similar to the paper above)
    # by contracting q into K
    S = einsum('rR,RZ->rZ',K,q)
    # Convert R into a matrix
    Rmat = R.ten.reshape((np.prod(R.shape[:3]),-1))
    # Convert S into a vector
    Svec = S.ten.reshape((-1))
    # Solve R*p=S
    pvec,_,_,_ = np.linalg.lstsq(Rmat,Svec,rcond=None)
    # Put p back into a symtensor
    p.ten = pvec.reshape(S.shape)
    # Evaluate the results
    cost = calc_cost_pq_lb(N,K,p,q)
    # Return result
    return p,cost

def optimize_q_lb(N,K,p,q):
    """
    Use least squares procedure to determine the q tensor
    """
    # Calculate R (similar to https://arxiv.org/pdf/1503.05345.pdf)
    # by contracting p into N
    R = einsum('rRyY,rz->zRyY',N,p)
    R = einsum('rRyY,yz->rRzY',R,p)
    # Calculate S (again similar to the paper above)
    # by contracting p into K
    S = einsum('rR,rz->zR',K,p)
    # Convert R into a matrix
    Rmat = R.ten.reshape((np.prod(R.shape[:3]),-1))
    # Convert S into a vector
    Svec = S.ten.reshape((-1))
    # Solve R*q=S
    qvec,_,_,_ = np.linalg.lstsq(Rmat,Svec,rcond=None)
    # Put q back into a symtensor
    q.ten = qvec.reshape(S.shape).transpose([1,2,0])
    # Evaluate the results
    cost = calc_cost_pq_lb(N,K,p,q)
    # Return result
    return q,cost

def do_als_lb(peps,eH,top,bot,left,right,mbd,
              maxiter=10,
              tol=1e-10,
              full_update=True,
              rand_init=False,
              stablesplit=True,
              truncate_loc=False,
              chi=100):
    """
    Do the alternating least squares procedure for hamiltonian acting on 
    left and bottom sites

    Note the location of the u,v,p,q tensors:

      +----+           +----+
      |    |-----------|    |
      +----+           +----+
         |                |
         u                |
         |                |
         v                |
         |                |
      +----+           +----+
      |    |--p----q---|    |
      +----+           +----+

    Args:
        peps : list of two lists of length 2
            The peps tensors in the unit cell with
            peps[0][0] -> bottom left tensor
            peps[0][1] -> top left tensor
            peps[1][0] -> bottom right tensor
            peps[1][1] -> top right tensor
        eH : list of three tensors
            The exponentiated MPO for the three sites 
        top : list of ndarrays
            The top environment tensors
        bot : list of ndarrays
            The bottom environment tensors
        left : list of ndarrays
            The left environment tensors
        right : list of ndarrays
            The right environment tensors
        mbd : int
            The maximum bond dimension

    Kwargs:
        maxiter : int
            The maximum number of ALS iterations to be performed
        tol : float
            The convergence tolerance for the ALS procedure
        full_update : bool
            Whether to actually do the full update (i.e. als) procedure
            or just split the time evolved tensors via svd
        rand_init : bool
            Whether to use a random initial guess (default is false, 
            meaning initial guess is determined via a simple update 
            style svd procedure)
        stablesplit : bool
            Whether to split the tensors after als procedure
            to equally split singular values. Default is True, 
            which adds minimal time and improves stability
        truncate_loc : bool
            Whether to truncate the environment tensors locally 
            when doing the ALS procedure
        chi : int
            The maximum bond dimension of any boundary mps
            If trunc_env == True, then this is the value that the 
            boundary MPS bond dimension will be reduced to

    Returns:
        peps : list of two lists of length 2
            The resulting peps tensors in the unit cell
    """
    # Check if this is a thermal state -------------------------------------------------
    thermal = len(peps[0][0].legs[2]) == 2

    # Generate initial guesses for the bond reducing tensors ---------------------------
    D = peps[0][0].shape[peps[0][0].legs[3][0]]
    Dh = eH[0].shape[eH[0].legs[2][0]]
    if rand_init:
        Dpeps = peps[0][0].shape[peps[0][0].legs[3][0]]
        Dham = eH[0].shape[2]
        chi = mbd
        u = rand_init_als_guess(Dpeps,Dham,chi)
        v = rand_init_als_guess(Dpeps,Dham,chi)
        p = rand_init_als_guess(Dpeps,Dham,chi)
        q = rand_init_als_guess(Dpeps,Dham,chi)
    else:
        u,v,p,q = su_init_als_guess_lb(peps,eH,mbd)

    # Determine initial cost function value -------------------------------------------
    cost_prev = None
    cost_prev_in = cost_prev

    # Begin ALS iterations ------------------------------------------------------------
    mpiprint(2,'\t\tStarting ALS'+'-'*20)
    if full_update:
        for i in range(maxiter):

            # Do ALS for the u and v tensors -------------------------------------------
            # Contract environments around u and v tensors
            N,K = calc_NK(peps,eH,top,bot,left,right,u,v,p,q,
                          lb=True,
                          pq=False,
                          chi=chi,
                          truncate=truncate_loc)
            # Do alternating least squares until convergence
            for j in range(maxiter):
                # Do optimization for u and v tensors
                u,cost = optimize_u_lb(N,K,u,v)
                mpiprint(2,"\t\tAfter u {}".format(cost))
                # Save iniital cost (in case this is the first cost calculation
                if cost_prev is None:
                    cost_prev = cost
                    cost_prev_in = cost
                v,cost = optimize_v_lb(N,K,u,v)
                mpiprint(2,"\t\tAfter v {}".format(cost))
                # Split u and v to share singular values equally
                if stablesplit:
                    uv = einsum('ab,cb->ac',u,v)
                    U,S,V = uv.svd(1,truncate_mbd=mbd,return_ent=False,return_wgt=False)
                    u = einsum('ab,bc->ac',U,S.sqrt())
                    v = einsum('ab,bc->ca',S.sqrt(),V)
                # Check for convergence
                if (abs(cost) < tol) or (abs((cost-cost_prev_in)/cost_prev) < tol):
                    cost_prev_in = cost
                    break
                elif j == maxiter-1:
                    cost_prev_in = cost
                else:
                    cost_prev_in = cost

            # Do ALS for the p and q tensors ------------------------------------------
            # Contract environments around p and q tensors
            N,K = calc_NK(peps,eH,top,bot,left,right,u,v,p,q,
                          lb=True,
                          pq=True,
                          chi=chi,
                          truncate=truncate_loc)
            # Do alternating least squares until convergence
            for j in range(maxiter):
                # Do optimization for p and q tensors
                p,cost = optimize_p_lb(N,K,p,q)
                mpiprint(2,"\t\tAfter p {}".format(cost))
                q,cost = optimize_q_lb(N,K,p,q)
                mpiprint(2,"\t\tAfter q {}".format(cost))
                # Split p and q to share singular values equally
                if stablesplit:
                    pq = einsum('ab,cb->ac',p,q)
                    U,S,V = pq.svd(1,truncate_mbd=mbd,return_ent=False,return_wgt=False)
                    p = einsum('ab,bc->ac',U,S.sqrt())
                    q = einsum('ab,bc->ca',S.sqrt(),V)
                # Check for convergence
                if (abs(cost) < tol) or (abs((cost-cost_prev_in)/cost_prev) < tol):
                    cost_prev_in = cost
                    break
                elif j == maxiter-1:
                    cost_prev_in = cost
                else:
                    cost_prev_in = cost

            # Check for convergence between u/v and p/q results -----------------------
            if (abs(cost) < tol) or (abs((cost-cost_prev)/cost_prev) < tol):
                break
            elif i == maxiter-1:
                cost_prev = cost
            else:
                cost_prev = cost
    
    # Absorb all bond reducers into the peps tensors ----------------------------------
    Hbra = [[None,None],[None,None]]
    if thermal:
        peps[0][1].unmerge_ind(2)
        Hbra[0][1] = einsum('ldparu,pPx->ldxParu',peps[0][1],eH[0])
        Hbra[0][1].merge_inds([3,4])
        peps[0][1].merge_inds([2,3])
        peps[0][0].unmerge_ind(2)
        Hbra[0][0] = einsum('ldparu,xpPy->ldParyux',peps[0][0],eH[1])
        Hbra[0][0].merge_inds([2,3])
        peps[0][0].merge_inds([2,3])
        peps[1][0].unmerge_ind(2)
        Hbra[1][0] = einsum('ldparu,ypP->lydParu',peps[1][0],eH[2])
        Hbra[1][0].merge_inds([3,4])
        peps[1][0].merge_inds([2,3])
        Hbra[1][1] = peps[1][1].copy()
    else:
        Hbra[0][1] = einsum('ldpru,pPx->ldxPru',peps[0][1],eH[0]) # Top left site
        Hbra[0][0] = einsum('ldpru,xpPy->ldPryux',peps[0][0],eH[1]) # Bottom left site
        Hbra[1][0] = einsum('ldpru,ypP->lydPru',peps[1][0],eH[2]) # Bottom right site
        Hbra[1][1] = peps[1][1].copy()
    # Merge indices as needed
    Hbra[0][1].merge_inds([1,2])
    Hbra[0][0].merge_inds([3,4])
    Hbra[0][0].merge_inds([4,5])
    Hbra[1][0].merge_inds([0,1])
    # Contract with reducing tensors
    Hbra_red = [[None,None],[None,None]]
    Hbra_red[0][1] = einsum('ldpru,dD->lDpru',Hbra[0][1],u)
    Hbra_red[0][0] = einsum('ldpru,rR->ldpRu',Hbra[0][0],p)
    Hbra_red[0][0] = einsum('ldpru,uU->ldprU',Hbra_red[0][0],v)
    Hbra_red[1][0] = einsum('ldpru,lL->Ldpru',Hbra[1][0],q)
    Hbra_red[1][1] = Hbra[1][1].copy()

    # Do some scaling to prevent large values
    Hbra_red[0][0] /= Hbra_red[0][0].abs().max()
    Hbra_red[0][1] /= Hbra_red[0][1].abs().max()
    Hbra_red[1][0] /= Hbra_red[1][0].abs().max()
    Hbra_red[1][1] /= Hbra_red[1][1].abs().max()

    # Return result
    return Hbra_red

def fill_empty_top(peps,left,right):
    """
    When the top is empty, this will put dummy matrices in 
    the correct places
    """
    nlegs_mid = [len(left[3].legs[2]),
                 len(peps[0][1].legs[4]),
                 len(peps[0][1].legs[4]),
                 len(peps[1][1].legs[4]),
                 len(peps[1][1].legs[4]),
                 len(right[3].legs[2])]
    top = [ones((1,)*(1+nlegs_mid[0]+1),sym=None,backend=peps[0][0].backend,dtype=peps[0][0].dtype,legs=[[0],list(range(1,nlegs_mid[0]+1)),[nlegs_mid[0]+1]]),
           ones((1,)*(1+nlegs_mid[1]+1),sym=None,backend=peps[0][0].backend,dtype=peps[0][0].dtype,legs=[[0],list(range(1,nlegs_mid[1]+1)),[nlegs_mid[1]+1]]),
           ones((1,)*(1+nlegs_mid[2]+1),sym=None,backend=peps[0][0].backend,dtype=peps[0][0].dtype,legs=[[0],list(range(1,nlegs_mid[2]+1)),[nlegs_mid[2]+1]]),
           ones((1,)*(1+nlegs_mid[3]+1),sym=None,backend=peps[0][0].backend,dtype=peps[0][0].dtype,legs=[[0],list(range(1,nlegs_mid[3]+1)),[nlegs_mid[3]+1]]),
           ones((1,)*(1+nlegs_mid[4]+1),sym=None,backend=peps[0][0].backend,dtype=peps[0][0].dtype,legs=[[0],list(range(1,nlegs_mid[4]+1)),[nlegs_mid[4]+1]]),
           ones((1,)*(1+nlegs_mid[5]+1),sym=None,backend=peps[0][0].backend,dtype=peps[0][0].dtype,legs=[[0],list(range(1,nlegs_mid[5]+1)),[nlegs_mid[5]+1]])]
    return top

def fill_empty_bot(peps,left,right):
    """
    When the top is empty, this will put dummy matrices in 
    the correct places
    """
    nlegs_mid = [len(left[0].legs[0]),
                 len(peps[0][0].legs[1]),
                 len(peps[0][0].legs[1]),
                 len(peps[1][0].legs[1]),
                 len(peps[1][0].legs[1]),
                 len(right[0].legs[0])]
    bot = [ones((1,)*(1+nlegs_mid[0]+1),sym=None,backend=peps[0][0].backend,dtype=peps[0][0].dtype,legs=[[0],list(range(1,nlegs_mid[0]+1)),[nlegs_mid[0]+1]]),
           ones((1,)*(1+nlegs_mid[1]+1),sym=None,backend=peps[0][0].backend,dtype=peps[0][0].dtype,legs=[[0],list(range(1,nlegs_mid[1]+1)),[nlegs_mid[1]+1]]),
           ones((1,)*(1+nlegs_mid[2]+1),sym=None,backend=peps[0][0].backend,dtype=peps[0][0].dtype,legs=[[0],list(range(1,nlegs_mid[2]+1)),[nlegs_mid[2]+1]]),
           ones((1,)*(1+nlegs_mid[3]+1),sym=None,backend=peps[0][0].backend,dtype=peps[0][0].dtype,legs=[[0],list(range(1,nlegs_mid[3]+1)),[nlegs_mid[3]+1]]),
           ones((1,)*(1+nlegs_mid[4]+1),sym=None,backend=peps[0][0].backend,dtype=peps[0][0].dtype,legs=[[0],list(range(1,nlegs_mid[4]+1)),[nlegs_mid[4]+1]]),
           ones((1,)*(1+nlegs_mid[5]+1),sym=None,backend=peps[0][0].backend,dtype=peps[0][0].dtype,legs=[[0],list(range(1,nlegs_mid[5]+1)),[nlegs_mid[5]+1]])]
    return bot

def mirror_rt2bl(peps,top,bot,left,right):
    """
    Flip the right top tensors to become the bottom left tensors

    Args:
        peps : list of two lists of length 2
            The peps tensors in the unit cell with
            peps[0][0] -> bottom left tensor
            peps[0][1] -> top left tensor
            peps[1][0] -> bottom right tensor
            peps[1][1] -> top right tensor
        top : list of ndarrays
            The top environment tensors
        bot : list of ndarrays
            The bottom environment tensors
        left : list of ndarrays
            The left environment tensors
        right : list of ndarrays
            The right environment tensors

    Returns:
        peps : list of two lists of length 2
            The resulting flipped peps tensors in the unit cell
        top : list of ndarrays
            The resulting top environment for the flipped tensors
        bot : list of ndarrays
            The resulting bottom environment for the flipped tensors
        left : list of ndarrays
            The resulting left environment for the flipped tensors
        right : list of ndarrays
            The resulting right environment for the flipped tensors
    """
    # Replace top and bottoms with ones if needed ----
    if bot is None:
        bot = fill_empty_bot(peps,left,right)
    if top is None:
        top = fill_empty_top(peps,left,right)
    # Save all the old stuff
    old_peps = peps
    old_top = top
    old_bot = bot
    old_left = left
    old_right = right
    # Create lists to hold new stuff
    top   = [None,None,None,None,None,None]
    bot   = [None,None,None,None,None,None]
    left  = [None,None,None,None]
    right = [None,None,None,None]
    # Flip the peps so we can use the lb als routines
    peps = [[None,None],[None,None]]
    peps[0][0] = old_peps[1][1].copy().transpose([4,3,2,1,0])
    peps[0][1] = old_peps[0][1].copy().transpose([4,3,2,1,0])
    peps[1][0] = old_peps[1][0].copy().transpose([4,3,2,1,0])
    peps[1][1] = old_peps[0][0].copy().transpose([4,3,2,1,0])
    # Put the old right environment into the bottom environment
    bot[0] = old_top[5].copy().transpose([2,0,1])
    bot[1] = old_right[3].copy().transpose([2,1,0])
    bot[2] = old_right[2].copy().transpose([2,1,0])
    bot[3] = old_right[1].copy().transpose([2,1,0])
    bot[4] = old_right[0].copy().transpose([2,1,0])
    bot[5] = old_bot[5].copy().transpose([1,0,2])
    bot = MPS(bot)
    # Put the old top environment into the left environment
    left[0] = old_top[4].copy().transpose([2,1,0])
    left[1] = old_top[3].copy().transpose([2,1,0])
    left[2] = old_top[2].copy().transpose([2,1,0])
    left[3] = old_top[1].copy().transpose([2,1,0])
    left = MPS(left)
    # Put the old bottom environment into the right environment
    right[0] = old_bot[4].copy().transpose([2,1,0])
    right[1] = old_bot[3].copy().transpose([2,1,0])
    right[2] = old_bot[2].copy().transpose([2,1,0])
    right[3] = old_bot[1].copy().transpose([2,1,0])
    # Put the old left environment into the top environment
    top[0] = old_top[0].copy().transpose([0,2,1])
    top[1] = old_left[3].copy().transpose([2,1,0])
    top[2] = old_left[2].copy().transpose([2,1,0])
    top[3] = old_left[1].copy().transpose([2,1,0])
    top[4] = old_left[0].copy().transpose([2,1,0])
    top[5] = old_bot[0].copy().transpose([1,2,0])
    # Return flipped results
    return peps,top,bot,left,right

def mirror_bl2rt(peps):
    """
    Flip the bottom left tensors to become the right top tensors
    NOTE - Only flips the peps, unlike rt2bl, which flips
    the peps and all environment tensors

    Args:
        peps : list of two lists of length 2
            The peps tensors in the unit cell with
            peps[0][0] -> bottom left tensor
            peps[0][1] -> top left tensor
            peps[1][0] -> bottom right tensor
            peps[1][1] -> top right tensor

    Returns:
        peps : list of two lists of length 2
            The resulting (copies of originals) peps tensors in the unit cell
    """
    old_peps = peps
    peps = [[None,None],[None,None]]
    peps[0][0] = old_peps[1][1].copy().transpose([4,3,2,1,0])
    peps[0][1] = old_peps[0][1].copy().transpose([4,3,2,1,0])
    peps[1][0] = old_peps[1][0].copy().transpose([4,3,2,1,0])
    peps[1][1] = old_peps[0][0].copy().transpose([4,3,2,1,0])
    return peps

def do_als_rt(peps,eH,top,bot,left,right,mbd,
              maxiter=10,
              tol=1e-10,
              full_update=True,
              rand_init=False,
              stablesplit=True,
              truncate_loc=False,
              chi=100):
    """
    Do the alternating least squares procedure for hamiltonian acting on 
    left and bottom sites

    Args:
        peps : list of two lists of length 2
            The peps tensors in the unit cell with
            peps[0][0] -> bottom left tensor
            peps[0][1] -> top left tensor
            peps[1][0] -> bottom right tensor
            peps[1][1] -> top right tensor
        eH : list of three tensors
            The exponentiated MPO for the three sites 
        top : list of ndarrays
            The top environment tensors
        bot : list of ndarrays
            The bottom environment tensors
        left : list of ndarrays
            The left environment tensors
        right : list of ndarrays
            The right environment tensors
        mbd : int
            The maximum bond dimension

    Kwargs:
        maxiter : int
            The maximum number of ALS iterations to be performed
        tol : float
            The convergence tolerance for the ALS procedure
        full_update : bool
            Whether to actually do the full update (i.e. als) procedure
            or just split the time evolved tensors via svd
        rand_init : bool
            Whether to use a random initial guess (default is false, 
            meaning initial guess is determined via a simple update 
            style svd procedure)
        stablesplit : bool
            Whether to split the tensors after als procedure
            to equally split singular values. Default is True, 
            which adds minimal time and improves stability
        truncate_loc : bool
            Whether to truncate the environment tensors locally 
            when doing the ALS procedure
        chi : int
            The maximum bond dimension of any boundary mps
            If trunc_env == True, then this is the value that the 
            boundary MPS bond dimension will be reduced to

    Returns:
        peps : list of two lists of length 2
            The resulting peps tensors in the unit cell
    """
    # Flip the peps so we can use the lb als routines
    peps,top,bot,left,right = mirror_rt2bl(peps,top,bot,left,right)
    # Evolve peps now using the lb full update routines
    peps = do_als_lb(peps,eH,top,bot,left,right,mbd,
                     maxiter=maxiter,
                     tol=tol,
                     full_update=full_update,
                     rand_init=rand_init,
                     stablesplit=stablesplit,
                     truncate_loc=truncate_loc,
                     chi=chi)
    # Flip the peps back
    peps = mirror_bl2rt(peps)
    # Return result
    return peps

def do_als(peps,eH,top,bot,left,right,mbd,
           maxiter=10,
           tol=1e-10,
           lb=True,
           full_update=True,
           rand_init=False,
           stablesplit=True,
           truncate_loc=False,
           chi=100):
    """
    Do the alternating least squares procedure

    Args:
        peps : list of two lists of length 2
            The peps tensors in the unit cell with
            peps[0][0] -> bottom left tensor
            peps[0][1] -> top left tensor
            peps[1][0] -> bottom right tensor
            peps[1][1] -> top right tensor
        eH : list of three tensors
            The exponentiated MPO for the three sites 
        top : list of ndarrays
            The top environment tensors
        bot : list of ndarrays
            The bottom environment tensors
        left : list of ndarrays
            The left environment tensors
        right : list of ndarrays
            The right environment tensors
        mbd : int
            The maximum bond dimension

    Kwargs:
        maxiter : int
            The maximum number of ALS iterations to be performed
        tol : float
            The convergence tolerance for the ALS procedure
        lb : bool
            Whether the MPO acts on the left and bottom edges (True)
            or the right and top edges (False)
        full_update : bool
            Whether to actually do the full update (i.e. als) procedure
            or just split the time evolved tensors via svd
        rand_init : bool
            Whether to use a random initial guess (default is false, 
            meaning initial guess is determined via a simple update 
            style svd procedure)
        stablesplit : bool
            Whether to split the tensors after als procedure
            to equally split singular values. Default is True, 
            which adds minimal time and improves stability
        truncate_loc : bool
            Whether to truncate the environment tensors locally 
            when doing the ALS procedure
        chi : int
            The maximum bond dimension of any boundary mps
            If trunc_env == True, then this is the value that the 
            boundary MPS bond dimension will be reduced to

    Returns:
        peps : list of two lists of length 2
            The resulting peps tensors in the unit cell
    """
    if lb:
        res = do_als_lb(peps,eH,top,bot,left,right,mbd,
                        maxiter=maxiter,
                        tol=tol,
                        full_update=full_update,
                        rand_init=rand_init,
                        stablesplit=stablesplit,
                        truncate_loc=truncate_loc,
                        chi=chi)
        return res
    else:
        res = do_als_rt(peps,eH,top,bot,left,right,mbd,
                        maxiter=maxiter,
                        tol=tol,
                        full_update=full_update,
                        rand_init=rand_init,
                        stablesplit=stablesplit,
                        truncate_loc=truncate_loc,
                        chi=chi)
        return res

def make_equal_distance_twotens(ten1,ten2,mbd,vertical=True):
    """
    Multiply two nearest neighbor peps tensors
    together and resplit them so singular values are
    equally split between the two tensors

    Args:
        ten1 : ndarray
            The left (or bottom) tensor
        ten2 : ndarray
            The right (or top) tensor
        mbd : int
            The maximum bond dimension

    Kwargs:
        vertical : bool
            Whether this is a vertical tensor connection (top/bottom)
            or a horizontal tensor connection (left/right)

    Returns:
        ten1 : ndarray
            The resulting left (or bottom) tensor
        ten2 : ndarray
            The resulting right (or top) tensor
    """
    # Create copies of the tensors first
    ten1 = ten1.copy()
    ten2 = ten2.copy()
    # Flip tensors along horizontal bonds
    if not vertical:
        ten1 = ten1.transpose([4,0,2,1,3])
        ten2 = ten2.transpose([4,0,2,1,3])
    # Pull the physical index off each tensor
    ten1 = ten1.transpose([0,1,3,2,4])
    (ub,sb,vb) = ten1.svd(3,return_ent=False,return_wgt=False)
    phys_b = einsum('aA,APU->aPU',sb,vb)
    ten2 = ten2.transpose([1,2,0,3,4])
    (ut,st,vt) = ten2.svd(2,return_ent=False,return_wgt=False)
    phys_t = einsum('DPa,aA->DPA',ut,st)
    # combine the two reduced tensors
    theta = einsum('aPU,UQb->aPQb',phys_b,phys_t)
    # take svd of result and truncate d if needed
    (u,s,v) = theta.svd(2,truncate_mbd=mbd,return_ent=False,return_wgt=False)
    # normalize s
    normfact = s.backend.sqrt(einsum('ij,jk->',s,s))
    #s /= normfact
    # recombine the tensors
    phys_b = einsum('aPU,Uu->aPu',u,s.sqrt())
    phys_t = einsum('dD,DPa->dPa',s.sqrt(),v)
    ten1 = einsum('LDRa,aPU->LDPRU',ub,phys_b)
    ten2 = einsum('DPa,aLRU->LDPRU',phys_t,vt)
    # Flip back horizontally bonded tensors
    if not vertical:
        ten1 = ten1.transpose([1,3,2,4,0])
        ten2 = ten2.transpose([1,3,2,4,0])
    return ten1,ten2

def make_equal_distance(peps,mbd,niter=1):
    """
    Multiply nearest neighbor peps tensors together and resplit them
    so that the singular values are equally split between the two tensors

    Args:
        peps : list of two lists of length 2
            The peps tensors in the unit cell with
            peps[0][0] -> bottom left tensor
            peps[0][1] -> top left tensor
            peps[1][0] -> bottom right tensor
            peps[1][1] -> top right tensor
        mbd : int
            The maximum bond dimension

    Kwargs:
        niter : int
            The number of times to go through the svd
            procedure to split singular values (default is
            1)

    Returns:
        peps : list of two lists of length 2
            The resulting peps tensors in the unit cell with
            singular values split equally among them.
    """
    # Possibly repeat a few times (if this improves things)
    mpiprint(2,'\t\tMaking resulting tensors equal distance')
    for i in range(niter):
        # Make equal distance 00,01 --------------------------
        peps[0][0],peps[0][1] = make_equal_distance_twotens(peps[0][0],peps[0][1],mbd)
        # Make equal distance 00,10 -------------------------- 
        peps[0][0],peps[1][0] = make_equal_distance_twotens(peps[0][0],peps[1][0],mbd,vertical=False)
        # Make equal distance 10,11 -------------------------- 
        peps[1][0],peps[1][1] = make_equal_distance_twotens(peps[1][0],peps[1][1],mbd)
        # Make equal distance 00,10 -------------------------- 
        peps[0][1],peps[1][1] = make_equal_distance_twotens(peps[0][1],peps[1][1],mbd,vertical=False)
    # Return resulting tensors
    return peps

def tebd_step_single_col(peps_col1,peps_col2,step_size,
                         left_bmpo,right_bmpo,ham_col,mbd,hamind,
                         chi=100,als_iter=10,als_tol=1e-10,
                         lb=True,full_update=True,
                         rand_als_init=False,stablesplit=True,
                         truncate_loc=False,chiloc=100):
    """
    Perform a single time evolution step for a single set of neighboring PEPS columns

    Args:
        peps_col1 : List of tensors
            A list of peps tensors corresponding to the first (left) peps column
        peps_col2 : List of tensors
            A list of peps tensors corresponding to the second (right) peps column
        step_size : float
            The trotter step size, may be a list of
            step sizes
        left_bmpo : MPS Object
            The boundary MPO for the left side of the peps
        right_bmpo : MPS Object
            The boundary MPO for the right side of the peps
        ham_col : List
            The suzuki-trotter decomposition of the
            Hamiltonian terms for interactions between
            the two columns
        mbd : int
            The maximum bond dimension of the PEPS
        hamind : int
            The index to specify which of the
            three hamiltonians (that are in a list for application
            to this unit cell) will be applied.

    Kwargs:
        als_iter : int
            The maximum number of alternating least squares iterations
        als_tol : float
            The convergence tolerance for the alternating
            least squares procedure
        lb : bool
            Whether this is for the Hamiltonian terms acting on the left
            and right edges of the 2x2 PEPS unit (if True) or for the terms
            acting on the right and top edges of the unit cell (if False)
        full_update: bool
            Whether to do full update (True) or simple update (False) procedure
        rand_als_init : bool
            Whether to use a random initial guess when starting 
            the als proecudure. (default is false, 
            meaning initial guess is determined via a simple update 
            style svd procedure)
        stablesplit : bool
            Whether to split the tensors after als procedure
            to equally split singular values. Default is True, 
            which adds minimal time and improves stability
        truncate_loc : bool
            Whether to truncate the environment tensors locally 
            when doing the ALS procedure
        chi : int
            The maximum bond dimension fo the boundary mps for 
            all left and right and top and bottom environments,
            except within the 2x2 unit cell
        chiloc : int
            The maximum bond dimension of the boundary mps in the local environment.
            If truncate_loc == True, then this is the value that the 
            boundary MPS bond dimension will be reduced to

    Returns:
        E : float
            The summed energy evaluation for Hamiltonian terms on only these two columns
        peps : list of two lists
            A list of the two resulting peps column after the time evolution step
    """
    # Calculate top and bottom environments
    top_envs = calc_top_envs2([peps_col1,peps_col2],
                              left_bmpo,
                              right_bmpo,
                              truncate=True,
                              chi=chi)
    bot_envs = calc_bot_envs2([peps_col1,peps_col2],
                              left_bmpo,
                              right_bmpo,
                              truncate=True,
                              chi=chi)

    # Loop through rows in the column
    E = peps_col1[0].backend.zeros(len(ham_col),dtype=peps_col1[0].dtype)
    for row in range(len(ham_col)):
        mpiprint(2,'\tRow tebd col = {}'.format(row))
        # Take exponential of the MPO
        eH = exp_mpo(ham_col[row][hamind],-step_size)

        # Determine which top and bot envs to use
        if row == 0:
            if len(peps_col1) == 2:
                # Only two sites in column, use identity at top and bottom
                top_env_curr = None
                bot_env_curr = None
            else:
                # On the bottom row
                top_env_curr = top_envs[row+2]
                bot_env_curr = None
        elif row == len(peps_col1)-2:
            # On the top row
            top_env_curr = None
            bot_env_curr = bot_envs[row-1]
        else:
            top_env_curr = top_envs[row+2]
            bot_env_curr = bot_envs[row-1]

        # Apply the gate using als
        res  = do_als([[peps_col1[row],peps_col1[row+1]],[peps_col2[row],peps_col2[row+1]]],
                      eH,
                      top_env_curr,
                      bot_env_curr,
                      left_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                      right_bmpo[row*2,row*2+1,row*2+2,row*2+3],
                      mbd,
                      maxiter=als_iter,
                      tol=als_tol,
                      lb=lb,
                      full_update=full_update,
                      rand_init=rand_als_init,
                      stablesplit=stablesplit,
                      truncate_loc=truncate_loc,
                      chi=chiloc)

        # Combine and equally split the tensors (and scale to avoid precision errors)
        res = make_equal_distance(res,mbd)

        # Put results back into the peps columns
        peps_col1[row] = res[0][0]
        peps_col1[row+1] = res[0][1]
        peps_col2[row] = res[1][0]
        peps_col2[row+1] = res[1][1]
        
        # Evaluate energy and norm locally
        mpiprint(2,'\t\tCalculating energy')
        E[row] = calc_local_nn_op(row,
                                  [peps_col1,peps_col2],
                                  ham_col,
                                  left_bmpo,
                                  right_bmpo,
                                  bot_envs,
                                  top_envs,
                                  chi=chi)

        # Update top and bottom environments
        if row == 0: prev_env = None
        else: prev_env = bot_envs[row-1]
        bot_envs[row] = update_bot_env2(row,
                                        [peps_col1,peps_col2],
                                        [[i.copy() for i in peps_col1],[i.copy() for i in peps_col2]],
                                        left_bmpo[2*row],
                                        left_bmpo[2*row+1],
                                        right_bmpo[2*row],
                                        right_bmpo[2*row+1],
                                        prev_env,
                                        truncate=True,
                                        chi=chi)

    # Return the result
    return E.sum(),[peps_col1,peps_col2]

def tebd_step(peps,ham,step_size,mbd,hamind,
              chi=100,als_iter=10,als_tol=1e-10,
              lb=True,full_update=True,
              rand_als_init=False,stablesplit=True,
              truncate_loc=False,chiloc=100):
    """
    Perform a single TEBD time step
    
    Args:
        peps : PEPS object
            The PEPS object to be evolved.
        ham : 3D array
            The suzuki-trotter decomposition of the
            Hamiltonian. An example of how this is constructed
            for the ising transverse field model
            is found in /mpo/j1j2.py
        step_size : float
            The trotter step size, may be a list of
            step sizes
        mbd : int
            The maximum bond dimension of the PEPS
        hamind : int
            The index to specify which of the
            three hamiltonians (that are in a list for application
            to this unit cell) will be applied.

    Kwargs:
        als_iter : int
            The maximum number of alternating least squares iterations
        als_tol : float
            The convergence tolerance for the alternating
            least squares procedure
        full_update: bool
            Whether to do full update (True) or simple update (False) procedure
        rand_als_init : bool
            Whether to use a random initial guess when starting 
            the als proecudure. (default is false, 
            meaning initial guess is determined via a simple update 
            style svd procedure)
        stablesplit : bool
            Whether to split the tensors after als procedure
            to equally split singular values. Default is True, 
            which adds minimal time and improves stability
        truncate_loc : bool
            Whether to truncate the environment tensors locally 
            when doing the ALS procedure
        chi : int
            The maximum bond dimension fo the boundary mps for 
            all left and right and top and bottom environments,
            except within the 2x2 unit cell
        chiloc : int
            The maximum bond dimension of the boundary mps in the local environment.
            If truncate_loc == True, then this is the value that the 
            boundary MPS bond dimension will be reduced to

    Returns:
        E : float
            The energy of the final peps result
        peps : PEPS object
            The final PEPS as a PEPS object
    """
    # Figure out peps size
    (Nx,Ny) = peps.shape

    # Compute the boundary MPOs
    right_bmpo = calc_right_bound_mpo(peps, 0,chi=chi,return_all=True)
    left_bmpo = [None]*(Nx-1)
    ident_bmpo = identity_mps(len(right_bmpo[0]),
                              dtype=peps[0][0].dtype,
                              sym=(peps[0][0].sym is not None),
                              backend=peps.backend)

    # Loop through all columns
    E = peps.backend.zeros((len(ham)),dtype=peps[0][0].dtype)
    for col in range(Nx-1):
        mpiprint(2,'column {}'.format(col))
        # Take TEBD Step
        if col == 0:
            res = tebd_step_single_col(peps[col],
                                       peps[col+1],
                                       step_size,
                                       ident_bmpo,
                                       right_bmpo[col+1],
                                       ham[col],
                                       mbd,
                                       hamind,
                                       als_iter=als_iter,
                                       als_tol=als_tol,
                                       lb=lb,
                                       full_update=full_update,
                                       rand_als_init=rand_als_init,
                                       stablesplit=stablesplit,
                                       truncate_loc=truncate_loc,
                                       chiloc=chiloc)
        elif col == Nx-2:
            res = tebd_step_single_col(peps[col],
                                       peps[col+1],
                                       step_size,
                                       left_bmpo[col-1],
                                       ident_bmpo,
                                       ham[col],
                                       mbd,
                                       hamind,
                                       als_iter=als_iter,
                                       als_tol=als_tol,
                                       lb=lb,
                                       full_update=full_update,
                                       rand_als_init=rand_als_init,
                                       stablesplit=stablesplit,
                                       truncate_loc=truncate_loc,
                                       chiloc=chiloc)
        else:
            res = tebd_step_single_col(peps[col],
                                       peps[col+1],
                                       step_size,
                                       left_bmpo[col-1],
                                       right_bmpo[col+1],
                                       ham[col],
                                       mbd,
                                       hamind,
                                       als_iter=als_iter,
                                       als_tol=als_tol,
                                       lb=lb,
                                       full_update=full_update,
                                       rand_als_init=rand_als_init,
                                       stablesplit=stablesplit,
                                       truncate_loc=truncate_loc,
                                       chiloc=chiloc)
        E[col] = res[0]
        peps[col] = res[1][0]
        peps[col+1] = res[1][1]

        # Update left boundary tensors
        if col == 0:
            left_bmpo[col] = update_left_bound_mpo(peps[col], None, chi=chi)
        elif col != Nx-1:
            left_bmpo[col] = update_left_bound_mpo(peps[col], left_bmpo[col-1], chi=chi)

    # Return result
    E = peps.backend.sum(E)
    return E,peps

def tebd_steps(peps,ham,step_size,n_step,conv_tol,mbd,
               chi=100,als_iter=10,als_tol=1e-10,
               full_update=True,
               rand_als_init=False,stablesplit=True,
               truncate_loc=False,chiloc=100):
    """
    Run all of the tebd time steps
    
    Args:
        peps : PEPS object
            The PEPS object to be evolved.
        ham : 3D array
            The suzuki-trotter decomposition of the
            Hamiltonian. An example of how this is constructed
            for the ising transverse field model
            is found in /mpo/j1j2.py
        step_size : float
            The trotter step size, may be a list of
            step sizes
        n_step : int
            The number of steps to be taken for each
            trotter step size. 
        conv_tol : float
            The convergence tolerance (for imaginary tebd ground state calcs)
        mbd : int
            The maximum bond dimension of the PEPS

    Kwargs:
        chi : int
            The boundary mpo maximum bond dimension
        als_iter : int
            The maximum number of alternating least squares iterations
        als_tol : float
            The convergence tolerance for the alternating
            least squares procedure
        full_update: bool
            Whether to do full update (True) or simple update (False) procedure
        rand_als_init : bool
            Whether to use a random initial guess when starting 
            the als proecudure. (default is false, 
            meaning initial guess is determined via a simple update 
            style svd procedure)
        stablesplit : bool
            Whether to split the tensors after als procedure
            to equally split singular values. Default is True, 
            which adds minimal time and improves stability
        truncate_loc : bool
            Whether to truncate the environment tensors locally 
            when doing the ALS procedure
        chi : int
            The maximum bond dimension fo the boundary mps for 
            all left and right and top and bottom environments,
            except within the 2x2 unit cell
        chiloc : int
            The maximum bond dimension of the boundary mps in the local environment.
            If truncate_loc == True, then this is the value that the 
            boundary MPS bond dimension will be reduced to

    Returns:
        E : float
            The energy of the final peps result
        peps : PEPS object
            The final PEPS as a PEPS object
    """
    nSite = len(peps)*len(peps[0])

    # Compute Initial Energy
    mpiprint(3,'Calculation Initial Energy/site')
    Eprev = peps.calc_op(ham,chi=chi,nn=True)
    mpiprint(0,'Initial Energy/site = {}'.format(Eprev/nSite))

    # Do a single tebd step
    for iter_cnt in range(n_step):

        # Do TEBD Step with first MPO
        #     __             __
        #    |1 |           |  |
        #    |o_|           |__|
        #     |               
        #     |               
        #     |               
        #     |               
        #     |_             __
        #    ||2|           |3 |
        #    |o--------------o_|
        hamind = 0
        _,peps = tebd_step(peps,ham,step_size,mbd,hamind,
                           chi=chi,als_iter=als_iter,als_tol=als_tol,
                           lb=True,full_update=full_update,
                           rand_als_init=rand_als_init,stablesplit=stablesplit,
                           truncate_loc=truncate_loc,chiloc=chiloc)
        peps.normalize()
        # Do TEBD Step with Second MPO
        #     __             __
        #    |  |           | 1|
        #    |__|           |o_|
        #                    |
        #                    |
        #                    |
        #                    |
        #     __             |_
        #    | 3|           ||2|
        #    |_o-------------o_|
        peps.flip()
        hamind = 1
        _,peps = tebd_step(peps,ham,step_size,mbd,hamind,
                           chi=chi,als_iter=als_iter,als_tol=als_tol,
                           lb=True,full_update=full_update,
                           rand_als_init=rand_als_init,stablesplit=stablesplit,
                           truncate_loc=truncate_loc,chiloc=chiloc)
        peps.flip()
        peps.normalize()
        # Do TEBD Step with Third MPO
        #     __             __
        #    |  o-------------o|
        #    |1_|           |2||
        #                     |
        #                     |
        #                     |
        #                     |
        #     __             _|
        #    |  |           | o|
        #    |__|           |3_|
        hamind = 2
        _,peps = tebd_step(peps,ham,step_size,mbd,hamind,
                           chi=chi,als_iter=als_iter,als_tol=als_tol,
                           lb=False,full_update=full_update,
                           rand_als_init=rand_als_init,stablesplit=stablesplit,
                           truncate_loc=truncate_loc,chiloc=chiloc)
        peps.normalize()

        # Save PEPS
        peps.save()
        
        # Compute Resulting Energy
        E = peps.calc_op(ham,chi=chi,nn=True)

        # Check for convergence
        mpiprint(0,'Energy/site = {}'.format(E/nSite))
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
             chi=100,
             thermal=False,
             norm_tol=20,
             singleLayer=True,
             max_norm_iter=20,
             dtype=float_,
             step_size=[0.1,0.01,0.001],
             n_step=5,
             conv_tol=1e-5,
             als_iter=20,
             als_tol=1e-8,
             peps_fname=None,
             peps_fdir='./',
             full_update=True,
             rand_als_init=False,
             stablesplit=True,
             truncate_loc=False,
             chiloc=100):
    """
    Run the TEBD (Simple or Full) algorithm for a PEPS

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
            is found in /mpo/j1j2.py

    Kwargs:
        peps : PEPS object
            The initial guess for the PEPS. If this is not
            provided, then a random peps will be initialized.
            Note that the bond dimension D should be the same
            as the initial calculation bond dimension, since
            no initial bond reduction or increase
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
        n_step : int
            The number of steps to be taken for each
            trotter step size. If it is a list, then
            len(step_size) == len(n_step) and
            len(D) == len(n_step) must both be True.
        conv_tol : float
            The convergence tolerance
        als_iter : int
            The maximum number of alternating least squares iterations
        als_tol : float
            The convergence tolerance for the alternating
            least squares procedure
        peps_fname : str
            The name of the saved peps file
        peps_fdir : str
            The location where the peps will be saved
        full_update: bool
            Whether to do full update (True) or simple update (False) procedure
        rand_als_init : bool
            Whether to use a random initial guess when starting 
            the als proecudure. (default is false, 
            meaning initial guess is determined via a simple update 
            style svd procedure)
        stablesplit : bool
            Whether to split the tensors after als procedure
            to equally split singular values. Default is True, 
            which adds minimal time and improves stability
        truncate_loc : bool
            Whether to truncate the environment tensors locally 
            when doing the ALS procedure
        chi : int
            The maximum bond dimension fo the boundary mps for 
            all left and right and top and bottom environments,
            except within the 2x2 unit cell
        chiloc : int
            The maximum bond dimension of the boundary mps in the local environment.
            If truncate_loc == True, then this is the value that the 
            boundary MPS bond dimension will be reduced to

    Returns:
        E : float
            The energy of the resulting PEPS
        peps : PEPS object
            The time evolved PEPS object
    """
    t0 = time.time()
    mpiprint(0,'\n\nStarting TEBD Calculation')
    mpiprint(0,'#'*50)

    # Ensure the optimization parameters, the
    # bond dimension, trotter step size, and number
    # of trotter steps are compatable.
    if thermal:
        conv_tol = 0.
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
                    thermal=thermal,
                    backend=backend,
                    norm_tol=norm_tol,
                    singleLayer=singleLayer,
                    max_norm_iter=max_norm_iter,
                    dtype=dtype,
                    fname=peps_fname,
                    fdir=peps_fdir)

    # Absorb lambda tensors if canonical
    if peps.ltensors is not None:
        peps.absorb_lambdas()

    # Loop over all (bond dims/step sizes/number of steps)
    for Dind in range(len(D)):

        mpiprint(0,'\nFU Calculation for (D,chi,dt) = ({},{},{})'.format(D[Dind],chi[Dind],step_size[Dind]))

        # Do a tebd evolution for given step size
        E,peps = tebd_steps(peps,
                            ham,
                            step_size[Dind],
                            n_step[Dind],
                            conv_tol[Dind],
                            D[Dind],
                            chi = chi[Dind],
                            als_iter=als_iter,
                            als_tol=als_tol,
                            full_update=full_update,
                            rand_als_init=rand_als_init,
                            stablesplit=stablesplit,
                            truncate_loc=truncate_loc,
                            chiloc=chiloc)

        # Increase MBD if needed
        if (len(D)-1 > Dind) and (D[Dind+1] > D[Dind]):
            peps.increase_mbd(D[Dind+1],chi=chi[Dind+1])
            peps.normalize()


    # Print out results
    mpiprint(0,'\n\n'+'#'*50)
    mpiprint(0,'FU TEBD Complete')
    mpiprint(0,'-------------')
    mpiprint(0,'Total time = {} s'.format(time.time()-t0))
    mpiprint(0,'Per Site Energy = {}'.format(E/(Nx*Ny)))

    return E,peps
