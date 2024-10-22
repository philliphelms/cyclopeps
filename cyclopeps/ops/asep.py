"""
Suzuki-Trotter decomposition of the tilted
generator for the 2D simple exclusion process.
"""

from cyclopeps.tools.utils import *
from cyclopeps.ops.ops import OPS
from cyclopeps.tools.ops_tools import *
from numpy import float_,exp,ones,zeros

import collections
from cyclopeps.tools.gen_ten import einsum

def return_op(Nx,Ny,params,sym=None,backend='numpy'):
    """
    Return the operators

    Args:
        Nx : int
            Lattice size in the x direction
        Ny : int
            Lattixe size in the y direction
        params : 1D Array
            The parameters for the hamiltonian.
            Here, the first four entries are the
            bulk hopping rates:
                params[0] = jump right
                params[1] = jump left
                params[2] = jump up
                params[3] = jump down
            The next four are the insertion rates:
                params[4] = insert to the right (left boundary)
                params[5] = insert to the left (right boundary)
                params[6] = insert upwards (bottom boundary)
                params[7] = insert downards (top boundary)
            The next four are the removal rates:
                params[8] = remove to the right (right boundary)
                params[9] = remove to the left (left boundary)
                params[10]= remove upwards (top boundary)
                params[11]= remove downwards (bottom boundary)
            The last two are the biasing parameters:
                params[12]= Bias in the x-direction
                params[13]= Bias in the y-direction

    Returns:
        ops
    """
    # Collect useful operators
    ops = OPS(sym=sym,backend=backend)

    # Convert params to matrices
    params = val2mat_params(Nx,Ny,params)
    (jr,jl,ju,jd, cr,cl,cu,cd, dr,dl,du,dd, sx,sy) = params

    # Operators within columns
    columns = []
    for x in range(Nx):
        # Create operator for single column
        col_ops = []
        for y in range(Ny-1):
            # Create operator for intereaction between sites (y,y+1)
            op = make_op(y,
                         ju[x,:],
                         jd[x,:],
                         cu[x,:],
                         cd[x,:],
                         du[x,:],
                         dd[x,:],
                         sy[x,:],
                         ops)
            # Add to list of column operators
            col_ops.append(op)
        # Add column of operators to list of columns
        columns.append(col_ops)

    # Operators within Rows
    rows = []
    for y in range(Ny):
        # Create operator for single row
        row_ops = []
        for x in range(Nx-1):
            # Create operator for interaction between sites (x,x+1)
            op = make_op(x,
                         jr[:,y],
                         jl[:,y],
                         cr[:,y],
                         cl[:,y],
                         dr[:,y],
                         dl[:,y],
                         sx[:,y],
                         ops)
            # Add to list of row operators
            row_ops.append(op)
        # Add row of operators to list of rows
        rows.append(row_ops)

    # Return Results
    return [columns,rows]

def make_op(row,ju,jd,cu,cd,du,dd,sy,ops):
    """
    Interaction between sites (row,row+1)
    """
    # Collect needed ops
    Sp = ops.Sp
    Sm = ops.Sm
    v = ops.v
    n = ops.n
    I = ops.I

    # -----------------------------------------
    # Hopping between sites
    
    # Hop up
    if DEBUG: mpiprint(0,'{} Sp_{}*Sm_{} - {} n_{}*v_{}'.format(ju[row]*exp(sy[row]),row,row+1,ju[row],row,row+1))
    op  = ju[row]*exp(sy[row])*quick_op(Sp,Sm)
    op -= ju[row]*quick_op(n,v)
    # Hop down
    if DEBUG: mpiprint(0,'{} Sm_{}*Sp_{} - {} v_{}*n_{}'.format(jd[row+1]*exp(-sy[row+1]),row,row+1,jd[row+1],row,row+1))
    op += jd[row+1]*exp(-sy[row+1])*quick_op(Sm,Sp)
    op -= jd[row+1]*quick_op(v,n)

    # ----------------------------------------
    # Top Site creation/Annihilation

    # Destroy upwards
    if DEBUG: mpiprint(0,'{} Sp_{} - {} n_{}'.format(du[row+1]*exp(sy[row+1]),row+1,du[row+1],row+1))
    op += du[row+1]*exp(sy[row+1])*quick_op(I,Sp)
    op -= du[row+1]*quick_op(I,n)
    # Destroy downwards
    if DEBUG: mpiprint(0,'{} Sp_{} - {} n_{}'.format(dd[row+1]*exp(-sy[row+1]),row+1,dd[row+1],row+1))
    op += dd[row+1]*exp(-sy[row+1])*quick_op(I,Sp)
    op -= dd[row+1]*quick_op(I,n)
    # Create upwards
    if DEBUG: mpiprint(0,'{} Sm_{} - {} v_{}'.format(cu[row+1]*exp(sy[row+1]),row+1,cu[row+1],row+1))
    op += cu[row+1]*exp(sy[row+1])*quick_op(I,Sm)
    op -= cu[row+1]*quick_op(I,v)
    # Create Downwards
    if DEBUG: mpiprint(0,'{} Sm_{} - {} v_{}'.format(cd[row+1]*exp(-sy[row+1]),row+1,cd[row+1],row+1))
    op += cd[row+1]*exp(-sy[row+1])*quick_op(I,Sm)
    op -= cd[row+1]*quick_op(I,v)

    if row == 0:
        # ----------------------------------------
        # Bottom Site Creation/Annihilation

        # Destroy upwards
        if DEBUG: mpiprint(0,'{} Sp_{} - {} n_{}'.format(du[row]*exp(sy[row]),row,du[row],row))
        op += du[row]*exp(sy[row])*quick_op(Sp,I)
        op -= du[row]*quick_op(n,I)
        # Destroy downwards
        if DEBUG: mpiprint(0,'{} Sp_{} - {} n_{}'.format(dd[row]*exp(-sy[row]),row,dd[row],row))
        op += dd[row]*exp(-sy[row])*quick_op(Sp,I)
        op -= dd[row]*quick_op(n,I)
        # Create upwards
        if DEBUG: mpiprint(0,'{} Sm_{} - {} v_{}'.format(cu[row]*exp(sy[row]),row,cu[row],row))
        op += cu[row]*exp(sy[row])*quick_op(Sm,I)
        op -= cu[row]*quick_op(v,I)
        # Create Downwards
        if DEBUG: mpiprint(0,'{} Sm_{} - {} v_{}'.format(cd[row]*exp(-sy[row]),row,cd[row],row))
        op += cd[row]*exp(-sy[row])*quick_op(Sm,I)
        op -= cd[row]*quick_op(v,I)

    # Return Result
    op *= -1
    return op


# Current operators ------------------------------------------------------------------------------
def return_curr_op(Nx,Ny,params,include_edges=False,sym=None,backend='numpy'):
    """
    Return the operators to calculate the current

    Args:
        Nx : int
            Lattice size in the x direction
        Ny : int
            Lattixe size in the y direction
        params : 1D Array
            The parameters for the hamiltonian.
            Here, the first four entries are the
            bulk hopping rates:
                params[0] = jump right
                params[1] = jump left
                params[2] = jump up
                params[3] = jump down
            The next four are the insertion rates:
                params[4] = insert to the right (left boundary)
                params[5] = insert to the left (right boundary)
                params[6] = insert upwards (bottom boundary)
                params[7] = insert downards (top boundary)
            The next four are the removal rates:
                params[8] = remove to the right (right boundary)
                params[9] = remove to the left (left boundary)
                params[10]= remove upwards (top boundary)
                params[11]= remove downwards (bottom boundary)
            The last two are the biasing parameters:
                params[12]= Bias in the x-direction
                params[13]= Bias in the y-direction

    Returns:
        ops
    """
    # Collect useful operators
    ops = OPS(sym=sym,backend=backend)

    # Convert params to matrices
    params = val2mat_params(Nx,Ny,params)
    (jr,jl,ju,jd, cr,cl,cu,cd, dr,dl,du,dd, sx,sy) = params

    # Operators within columns
    columns = []
    for x in range(Nx):
        # Create operator for single column
        col_ops = []
        for y in range(Ny-1):
            # Create operator for intereaction between sites (y,y+1)
            op = make_curr_op(y,
                              ju[x,:],
                              jd[x,:],
                              cu[x,:],
                              cd[x,:],
                              du[x,:],
                              dd[x,:],
                              sy[x,:],
                              ops,
                              include_edges=include_edges)
            # Add to list of column operators
            col_ops.append(op)
        # Add column of operators to list of columns
        columns.append(col_ops)

    # Operators within Rows
    rows = []
    for y in range(Ny):
        # Create operator for single row
        row_ops = []
        for x in range(Nx-1):
            # Create operator for interaction between sites (x,x+1)
            op = make_curr_op(x,
                              jr[:,y],
                              jl[:,y],
                              cr[:,y],
                              cl[:,y],
                              dr[:,y],
                              dl[:,y],
                              sx[:,y],
                              ops,
                              include_edges=include_edges)
            # Add to list of row operators
            row_ops.append(op)
        # Add row of operators to list of rows
        rows.append(row_ops)

    # Return Results
    return [columns,rows]

def make_curr_op(row,ju,jd,cu,cd,du,dd,sy,ops,include_edges=False):
    """
    Interaction between sites (row,row+1)
    """
    # Collect needed operators
    Sp = ops.Sp
    Sm = ops.Sm
    v = ops.v
    n = ops.n
    I = ops.I

    # -----------------------------------------
    # Hopping between sites
    # Hop up
    if DEBUG: mpiprint(0,'{} Sp_{}*Sm_{}'.format(ju[row]*exp(sy[row]),row,row+1))
    op  = ju[row]*exp(sy[row])*quick_op(Sp,Sm)
    # Hop down
    if DEBUG: mpiprint(0,'{} Sm_{}*Sp_{}'.format(jd[row]*exp(-sy[row]),row,row+1))
    op -= jd[row+1]*exp(-sy[row+1])*quick_op(Sm,Sp)

    if include_edges:
        # ----------------------------------------
        # Top Site creation/Annihilation

        # Destroy upwards
        if DEBUG: mpiprint(0,'{} Sp_{}'.format(du[row+1]*exp(sy[row+1]),row+1))
        op += du[row+1]*exp(sy[row+1])*quick_op(I,Sp)
        # Destroy downwards
        if DEBUG: mpiprint(0,'{} Sp_{}'.format(dd[row+1]*exp(-sy[row+1]),row+1))
        op -= dd[row+1]*exp(-sy[row+1])*quick_op(I,Sp)
        # Create upwards
        if DEBUG: mpiprint(0,'{} Sm_{}'.format(cu[row+1]*exp(sy[row+1]),row+1))
        op += cu[row+1]*exp(sy[row+1])*quick_op(I,Sm)
        # Create Downwards
        if DEBUG: mpiprint(0,'{} Sm_{}'.format(cd[row+1]*exp(-sy[row+1]),row+1))
        op -= cd[row+1]*exp(-sy[row+1])*quick_op(I,Sm)

        if row == 0:
            # ----------------------------------------
            # Bottom Site Creation/Annihilation

            # Destroy upwards
            if DEBUG: mpiprint(0,'{} Sp_{}'.format(du[row]*exp(sy[row]),row))
            op += du[row]*exp(sy[row])*quick_op(Sp,I)
            # Destroy downwards
            if DEBUG: mpiprint(0,'{} Sp_{}'.format(dd[row]*exp(-sy[row]),row))
            op -= dd[row]*exp(-sy[row])*quick_op(Sp,I)
            # Create upwards
            if DEBUG: mpiprint(0,'{} Sm_{}'.format(cu[row]*exp(sy[row]),row))
            op += cu[row]*exp(sy[row])*quick_op(Sm,I)
            # Create Downwards
            if DEBUG: mpiprint(0,'{} Sm_{}'.format(cd[row]*exp(-sy[row]),row))
            op -= cd[row]*exp(-sy[row])*quick_op(Sm,I)

    # Return Result
    op *= -1
    return op

# Helpful Functions ------------------------------------------------------------------------------
def val2mat_params(Nx,Ny,params):
    # Extract values
    jr = params[0]
    jl = params[1]
    ju = params[2]
    jd = params[3]
    cr = params[4]
    cl = params[5]
    cu = params[6]
    cd = params[7]
    dr = params[8]
    dl = params[9]
    du = params[10]
    dd = params[11]
    sx = params[12]
    sy = params[13]

    # Bulk Hopping Rates
    if not isinstance(jr,(collections.Sequence)):
        jr_m = jr*ones((Nx,Ny),dtype=float_)
    else: jr_m = jr
    if not isinstance(jl,(collections.Sequence)):
        jl_m = jl*ones((Nx,Ny),dtype=float_)
    else: jl_m = jl
    if not isinstance(ju,(collections.Sequence)):
        ju_m = ju*ones((Nx,Ny),dtype=float_)
    else: ju_m = ju
    if not isinstance(jd,(collections.Sequence)):
        jd_m = jd*ones((Nx,Ny),dtype=float_)
    else: jd_m = jd

    # Insertion rates
    if not isinstance(cr,(collections.Sequence)):
        cr_m = zeros((Nx,Ny),dtype=float_)
        cr_m[0,:] = cr
    if not isinstance(cl,(collections.Sequence)):
        cl_m = zeros((Nx,Ny),dtype=float_)
        cl_m[Nx-1,:] = cl
    if not isinstance(cu,(collections.Sequence)):
        cu_m = zeros((Nx,Ny),dtype=float_)
        cu_m[:,Ny-1] = cu
    if not isinstance(cd,(collections.Sequence)):
        cd_m = zeros((Nx,Ny),dtype=float_)
        cd_m[:,0] = cd

    # Removal Rates
    if not isinstance(dr,(collections.Sequence)):
        dr_m = zeros((Nx,Ny),dtype=float_)
        dr_m[Nx-1,:] = dr
    if not isinstance(dl,(collections.Sequence)):
        dl_m = zeros((Nx,Ny),dtype=float_)
        dl_m[0,:] = dl
    if not isinstance(du,(collections.Sequence)):
        du_m = zeros((Nx,Ny),dtype=float_)
        du_m[:,0] = du
    if not isinstance(dd,(collections.Sequence)):
        dd_m = zeros((Nx,Ny),dtype=float_)
        dd_m[:,Ny-1] = dd

    # Biasing Terms
    if not isinstance(sx,(collections.Sequence)):
        sx_m = sx*ones((Nx,Ny),dtype=float_)
    else: sx_m = sx
    if not isinstance(sy,(collections.Sequence)):
        sy_m = sy*ones((Nx,Ny),dtype=float_)
    else: sy_m = sy

    # Return Result
    return (jr_m,jl_m,ju_m,jd_m,cr_m,cl_m,cu_m,cd_m,dr_m,dl_m,du_m,dd_m,sx_m,sy_m)
