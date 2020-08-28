"""
A wrapper for generic and symmetric tensors
providing required functionality for PEPS calculations

Author: Phillip Helms <phelms@caltech.edu>
Date: January 2020

"""
from numpy import float_
from cyclopeps.tools.utils import *
try:
    import symtensor.sym as symlib
    from symtensor.tools.la import symqr, symsvd
except:
    symlib,symqr,symsvd = None,None,None
import copy
import itertools
import sys
import numpy as np

LETTERS = 'abcdefghijklmnopqrstuvwxyz'
FLIP = {'+':'-','-':'+'}

###########################################################
# Functions
###########################################################
def calc_entanglement(S,backend=np):
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
    S = S.copy()
    # Ensure correct normalization
    norm_fact = backend.dot(S,S.conj())**(1./2.)+1e-100
    S /= norm_fact

    # Calc Entanglement Spectrum
    S += 1e-100
    EEspec = -S*S.conj()*backend.log2(S*S.conj())

    # Sum to get Entanglement Entropy
    EE = backend.sum(EEspec)

    # Return Results
    return EE,EEspec

def qr_ten(ten,split_ind,rq=False,backend=np.linalg):
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
    _ten = ten.copy()
    _ten = _ten.reshape([int(np.prod(ten_shape[:split_ind])),int(np.prod(ten_shape[split_ind:]))])

    # Perform svd
    mpiprint(9,'Perform actual qr')
    if not rq:
        # Do the QR Decomposition
        Q,R = backend.qr(_ten)

        # Reshape to match correct tensor format
        mpiprint(9,'Reshape to match original tensor dimensions')
        new_dims = ten_shape[:split_ind]+(int(np.prod(Q.shape)/np.prod(ten_shape[:split_ind])),)
        Q = Q.reshape(new_dims)
        new_dims = (int(np.prod(R.shape)/np.prod(ten_shape[split_ind:])),)+ten_shape[split_ind:]
        R = R.reshape(new_dims)

        # Quick check to see if it worked
        subscripts = LETTERS[:len(Q.shape)]+','+\
                     LETTERS[len(Q.shape)-1:len(Q.shape)-1+len(R.shape)]+'->'+\
                     LETTERS[:len(Q.shape)-1]+LETTERS[len(Q.shape):len(Q.shape)-1+len(R.shape)]
        #assert(np.allclose(ten,backend.einsum(subscripts,Q,R),rtol=1e-6))
    else:
        raise NotImplementedError()
    # Return results
    if rq:
        return R,Q
    else:
        return Q,R

def svd_ten(ten,split_ind,truncate_mbd=1e100,return_ent=True,return_wgt=True,backend=np.linalg):
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
    ten = ten.reshape([int(np.prod(ten_shape[:split_ind])),int(np.prod(ten_shape[split_ind:]))])

    # Perform svd
    mpiprint(9,'Perform actual svd')
    U,S,V = backend.svd(ten)

    # Compute Entanglement
    mpiprint(9,'Calculate the entanglment')
    if return_ent or return_wgt:
        EE,EEs = calc_entanglement(S,backend=backend)

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
    S = backend.diag(S)
    V = V[:D,:]

    # Reshape to match correct tensor format
    mpiprint(9,'Reshape to match original tensor dimensions')
    new_dims = ten_shape[:split_ind]+(int(np.prod(U.shape)/np.prod(ten_shape[:split_ind])),)
    U = U.reshape(new_dims)
    new_dims = (int(np.prod(V.shape)/np.prod(ten_shape[split_ind:])),)+ten_shape[split_ind:]
    V = V.reshape(new_dims)

    # Print some results
    if return_ent or return_wgt:
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

def eye(D,Z,is_symmetric=False,backend='numpy',dtype=float_,legs=None):
    """
    Create an identity tensor

    Args:
        D : int
            The size of each quantum number sector
        Z : int
            The number of quantum number sectors

    Kwargs:
        backend : str
            A string indicating the backend to be used
            for tensor creation and operations. 
            Options include:
            'ctf'      - a ctf tensor
            'numpy'    - a numpy tensor
        dtype : dtype
            The data type for the tensor, i.e. np.float_,np.complex128,etc.
    """
    if isinstance(backend,str):
        backend = load_lib(backend)
    if not is_symmetric:
        # Create a dense tensor
        ten = backend.eye(D,dtype=dtype)
        ten = GEN_TEN(ten=ten,backend=backend,legs=legs)
        return ten
    else:
        # Create a symmetric tensor
        sym = ['+-',[Z,Z],None,None]
        #if order == '+':
        #    sym = ['+-',[Z,Z],None,None]
        #else:
        #    sym = ['-+',[Z,Z],None,None]
        ten = GEN_TEN(shape=(D,D),sym=sym,backend=backend,dtype=dtype,legs=legs)
        for i in range(ten.ten.array.shape[0]):
            ten.ten.array[i,:,:] = backend.eye(ten.ten.array.shape[1])
        return ten

def rand(shape,sym=None,backend='numpy',dtype=float_,legs=None):
    """
    Create a random gen_ten tensor

    Args:
        shape : tuple
            The dimensions of each leg of the random tensor created

    Kwargs:
        sym : 
            If None, then a non-symmetric tensor will be created (default).
            Otherwise, this will create a symmetric tensor. 
            sym should be a list of length four with:
            sym[0] -> str of signs, i.e. '++--', for direction of symmetry arrows
            sym[1] -> list of ranges, i.e. [range(2)]*4, for number of sectors per tensor leg
            sym[2] -> integer, i.e. 0, for net value
            sym[3] -> integer, i.e. 2, for modulo values in the symmetry
        backend : str
            A string indicating the backend to be used
            for tensor creation and operations. 
            Options include:
            'ctf'      - a ctf tensor
            'numpy'    - a numpy tensor
        dtype : dtype
            The data type for the tensor, i.e. np.float_,np.complex128,etc.
    """
    if sym is not None:
        sym = [(sym[0]+',')[:-1],sym[1],sym[2],sym[3]]
    ten = GEN_TEN(shape=shape,sym=sym,backend=backend,dtype=dtype,legs=legs)
    ten.randomize()
    return ten

def ones(shape,sym=None,backend='numpy',dtype=float_,legs=None):
    """
    Create a gen_ten tensor filled with ones

    Args:
        shape : tuple
            The dimensions of each leg of the random tensor created

    Kwargs:
        sym : 
            If None, then a non-symmetric tensor will be created (default).
            Otherwise, this will create a symmetric tensor. 
            sym should be a list of length four with:
            sym[0] -> str of signs, i.e. '++--', for direction of symmetry arrows
            sym[1] -> list of ranges, i.e. [range(2)]*4, for number of sectors per tensor leg
            sym[2] -> integer, i.e. 0, for net value
            sym[3] -> integer, i.e. 2, for modulo values in the symmetry
        backend : str
            A string indicating the backend to be used
            for tensor creation and operations. 
            Options include:
            'ctf'      - a ctf tensor
            'numpy'    - a numpy tensor
        dtype : dtype
            The data type for the tensor, i.e. np.float_,np.complex128,etc.
    """
    if sym is not None:
        sym = [(sym[0]+',')[:-1],sym[1],sym[2],sym[3]]
    ten = GEN_TEN(shape=shape,sym=sym,backend=backend,dtype=dtype,legs=legs)
    ten.fill_all(1.)
    return ten

def zeros(shape,sym=None,backend='numpy',dtype=float_,legs=None):
    """
    Create a gen_ten tensor filled with zeros

    Args:
        shape : tuple
            The dimensions of each leg of the random tensor created

    Kwargs:
        sym : 
            If None, then a non-symmetric tensor will be created (default).
            Otherwise, this will create a symmetric tensor. 
            sym should be a list of length four with:
            sym[0] -> str of signs, i.e. '++--', for direction of symmetry arrows
            sym[1] -> list of ranges, i.e. [range(2)]*4, for number of sectors per tensor leg
            sym[2] -> integer, i.e. 0, for net value
            sym[3] -> integer, i.e. 2, for modulo values in the symmetry
        backend : str
            A string indicating the backend to be used
            for tensor creation and operations. 
            Options include:
            'ctf'      - a ctf tensor
            'numpy'    - a numpy tensor
        dtype : dtype
            The data type for the tensor, i.e. np.float_,np.complex128,etc.
    """
    if sym is not None:
        sym = [(sym[0]+',')[:-1],sym[1],sym[2],sym[3]]
    ten = GEN_TEN(shape=shape,sym=sym,backend=backend,dtype=dtype,legs=legs)
    ten.fill_all(0.)
    return ten

def find_all(s,ch):
    """
    Find all instances of a character in a string
    """
    return [i for i, ltr in enumerate(s) if ltr == ch]

LETTERS = 'abcdefghijklmnoprstuvwxyz'

def replace_caps(subscripts):
    """
    Replace all capital letters in a einsum subscript
    """
    unique_chars = list(set(list(subscripts)))
    for i in range(len(unique_chars)):
        if (unique_chars[i].isupper()) or (unique_chars[i] == 'q'):
            # Find all instances of that character
            charinds = find_all(subscripts,unique_chars[i])
            # find a letter to replace it
            for j in range(len(LETTERS)):
                if subscripts.find(LETTERS[j]) == -1:
                    subscripts = subscripts.replace(unique_chars[i],LETTERS[j])
                    break
                elif j == 25:
                    mpiprint(0,'There are no more strings left!')
                    sys.exit()
    return subscripts

def unmerge_subscripts(subscripts,Alegs,Blegs):
    """
    When doing einsum with merged bonds, this will add additional
    subscripts for the legs that have been merged into one
    """
    unique_chars = list(set(list(subscripts)))
    all_unique_chars = ''.join(list(set(list(subscripts))))
    unique_chars.remove('-')
    unique_chars.remove('>')
    unique_chars.remove(',')
    [strin,strout] = subscripts.split('->')
    [strA,strB] = strin.split(',')
    _strA,_strB,_strout = strA,strB,strout
    for i in range(len(unique_chars)):
        # Find how many entries are in that tensor
        strA_locs = _strA.find(unique_chars[i])
        strB_locs = _strB.find(unique_chars[i])
        strout_locs = _strout.find(unique_chars[i])
        if not (strA_locs == -1):
            nlegscomb = len(Alegs[strA_locs])
        if not (strB_locs == -1):
            nlegscomb = len(Blegs[strB_locs])
        if nlegscomb > 1:

            replacement_letters = unique_chars[i]
            for j in range(1,nlegscomb):
                for k in range(len(LETTERS)):
                    if (all_unique_chars.find(LETTERS[k]) == -1):
                        replacement_letters += LETTERS[k]
                        all_unique_chars += LETTERS[k]
                        break
                    elif k == 25:
                        mpiprint(0,'There are no more letters left!')
                        sys.exit()
            if not (strA_locs == -1):
                strA = strA.replace(unique_chars[i],replacement_letters)
            if not (strB_locs == -1):
                strB = strB.replace(unique_chars[i],replacement_letters)
            if not (strout_locs == -1):
                strout = strout.replace(unique_chars[i],replacement_letters)

    subscripts = strA+','+strB+'->'+strout
    return subscripts

def einsum(subscripts,opA,opB):
    """
    An einsum for contraction between two general tensors. 
    The only critical addition is the ability to deal with merged indices

    Args:
        indstr : string
            Specifies the subscripts for summation as comma separated list of subscript labels
        opA : array_like
            The first array for contraction
        opB : array_like
            The second array for contraction

    Returns:
        output : ndarray
            The resulting array from the einsum calculation
    """
    print_str = ''
    # Format String ------------------------------------
    _subscripts = subscripts
    subscripts = replace_caps(subscripts)
    subscripts = unmerge_subscripts(subscripts,opA.legs,opB.legs)
    # Do einsum
    try:
        res = opA.lib.einsum(subscripts,opA.ten,opB.ten)
    except Exception as e:
        if not (opA.lib == opB.lib):
            raise ValueError("Backends do not match for the two tensors")
        if opB.sym is None:
            print('{},{},{},{},{}'.format(subscripts,opA.ten.shape,opB.ten.shape,opA.lib,opB.lib))
        else:
            print('{},{},{},{},{}'.format(subscripts,opA.ten.array.shape,opB.ten.array.shape,type(opA.ten.array),type(opB.ten.array)))
        res = opA.lib.einsum(subscripts,opA.ten,opB.ten)
    # Create a new gen_ten (with correctly lumped legs) from the result
    # Find resulting sym
    if opA.sym is not None:
        if hasattr(res,'sym'):
            sym = res.sym
            sym = [(sym[0]+',')[:-1],sym[1],sym[2],sym[3]]
        else:
            # Constant returned (don't need to put into gen_ten)
            sym = None
            #return res
    else:
        sym = None
    # Find resulting legs
    [strin,strout] = _subscripts.split('->')
    [strA,strB] = strin.split(',')
    legs = []
    cnt = 0
    for i in range(len(strout)):
        strA_loc = strA.find(strout[i])
        strB_loc = strB.find(strout[i])
        if not (strA_loc == -1):
            legs += [list(range(cnt,cnt+len(opA.legs[strA_loc])))]
            cnt += len(opA.legs[strA_loc])
        else:
            legs += [list(range(cnt,cnt+len(opB.legs[strB_loc])))]
            cnt += len(opB.legs[strB_loc])
        #print('\t\t\tcnt {}, legs {}'.format(cnt,legs))
    if not isinstance(res,float):
        if len(subscripts.split('->')[1]) == 0:
            # If sized 1 array, convert to float
            ind = (0,)*len(res.shape)
            res = res[ind]
        else:
            res = GEN_TEN(sym = sym,
                          backend = opA.backend,
                          ten = res,
                          legs = legs)
    return res

###########################################################
# Object
###########################################################
class GEN_TEN:
    """
    A generic tensor class
    """
    def __init__(self,shape=None,sym=None,backend='numpy',dtype=float_,ten=None,legs=None):
        """
        Create a tensor of zeros of the correct tensor type

        Args:
            shape : tuple
                The dimensions of each tensor leg.
                Note, if using symtensors, the full dimension
                of each leg will be multiplied by the range
                of that leg, i.e. Zn

        Kwargs:
            backend : str
                A string indicating the backend to be used
                for tensor creation and operations. 
                Options include:
                'ctf'      - a ctf tensor
                'numpy'    - a numpy tensor
            sym : bool
                If True, then a symtensor will be created, otherwise, 
                the tensor will have no symmetry
        dtype : dtype
            The data type for the tensor, i.e. np.float_,np.complex128,etc.
        """
        # Load Backend
        if ten is None:
            self.backend = backend
        elif sym is None:
            self.backend = backend
        else:
            self.backend = ten.backend
        self.backend = load_lib(self.backend)

        # Set Tensor dtype
        if ten is None:
            self.dtype   = dtype
        else:
            self.dtype   = ten.dtype

        # Set Symmetry
        if sym is not None:
            self.sym = [(sym[0]+',')[:-1],sym[1],sym[2],sym[3]]
        else:
            self.sym = None

        # Set actual tensor
        if ten is None:
            # Create a zero tensor
            if sym is None:
                self.ten = self.lib.zeros(shape,dtype=dtype)
            else:
                if symlib is None: raise ImportError("Symtensor module not found")
                self.ten = symlib.zeros(shape,sym=sym,backend=backend,dtype=dtype)
        else:
            self.ten = ten
            try:
                sym = self.ten.sym
                self.sym = [(sym[0]+',')[:-1],sym[1],sym[2],sym[3]]
            except:
                self.sym = None
                pass
        
        # Create combined index
        if legs is None:
            self.legs = [[i] for i in range(self.ten.ndim)]
        else:
            self.legs = legs
        self.nlegs = len(self.legs)

    @property
    def ndim(self):
        return len(self.legs)

    @property
    def lib(self):
        if self.sym is None:
            return self.backend
        else:
            if symlib is None: raise ImportError("Symtensor module not found")
            return symlib

    @property
    def shape(self):
        return self.ten.shape

    @property
    def full_shape(self):
        if self.sym is None:
            return self.ten.shape
        else:
            shape = list(self.ten.shape)
            for i in range(len(shape)):
                shape[i] *= len(self.ten.sym[1][i])
            return tuple(shape)

    @property
    def qn_sectors(self):
        if self.sym is None:
            return [range(1)]*len(self.shape)
        else:
            return self.sym[1]

    @property
    def is_symmetric(self):
        return (not (self.sym is None))

    def randomize(self):
        """
        Fill the tensor with random elements
        """
        self.ten[:] = 0.
        if self.sym is None:
            self.ten += self.backend.random(self.ten.shape)
        else:
            self.ten += self.backend.random(self.ten.array.shape)

    def fill_all(self,value):
        """
        Fill the tensor with a given value
        """
        self.ten[:] = 0.
        if self.sym is None:
            self.ten += value*self.backend.ones(self.ten.shape)
        else:
            self.ten += value*self.backend.ones(self.ten.array.shape)

    def make_sparse(self):
        """
        Convert the symmetric tensor into a sparse tensor
        """
        if not self.is_symmetric:
            return self.copy()
        else:
            # Make a copy of the tensor
            newten = self.copy()
            # Convert the tensor to a sparse one
            nind = len(newten.ten.shape)
            newten.ten = newten.ten.make_sparse()
            newten.sym = None
            # Reshape the resulting sparse tensor
            order = []
            newshape = []
            shape = newten.ten.shape
            for i in range(nind):
                order += [i,nind+i]
                newshape += [shape[i]*shape[nind+i]]
            newten.ten = newten.backend.transpose(newten.ten,order)
            newten.ten = newten.backend.reshape(newten.ten,newshape)
            return newten

    def copy(self):
        """
        Return a copy of the gen_ten object
        """
        return self._as_new_tensor(self.ten.copy())

    def _as_new_tensor(self,ten):
        newten = GEN_TEN(ten=ten.copy(),
                         backend=self.backend,
                         legs=copy.deepcopy(self.legs))
        return newten

    def __str__(self):
        if self.sym:
            return self.ten.array.__str__()
        else:
            return self.ten.__str__()

    def transpose(self,axes):
        """
        Transpose the tensor
        """
        # Get actual transpose indices
        _axes = [self.legs[i] for i in axes]
        _axes = list(itertools.chain(*_axes))
        newten = self.ten.transpose(*_axes)
        # Update legs
        newlegs = []
        ind = 0
        for i in range(len(axes)):
            newlegs.append(list(range(ind,ind+len(self.legs[axes[i]]))))
            ind += len(self.legs[axes[i]])
        return GEN_TEN(ten=newten,backend=self.backend,legs=newlegs)

    def remove_empty_ind(self,ind):
        """
        Remove an index of size 1
        """
        init_shape = self.ten.shape
        # Check that we are summing over only one index
        for i in range(len(self.legs[ind])):
            assert(init_shape[self.legs[ind][i]] == 1)
        # Separate cases for tensors with and without symmetry
        if self.sym is None:
            # Sum over the single index for tensor without symmetry
            newten = self.ten.copy()
            for i in range(len(self.legs[ind]))[::-1]:
                newten = newten.sum(self.legs[ind][i])
        else:
            # More complex for symtensors
            sym = self.sym
            if ind == len(self.legs)-1:
                # Number of indices to be removed
                ntrunc = len(self.legs[-1])
                # Transpose tensor (so removed inds are in fromt
                neworder = [self.legs[ind]]+self.legs[0:ind]
                neworder = [item for sublist in neworder for item in sublist]
                newten = self.ten.copy()
                newten = newten.transpose(*neworder)
                newten = newten.array.copy()
                for i in range(ntrunc):
                    assert(newten.shape[i+len(sym[0])-1] == 1)
                # Sum over correct legs
                for i in range(ntrunc):
                    newten = newten.sum(i+len(sym[0])-1)
                for i in range(ntrunc):
                    newten = newten.sum(i)
                # Adjust Symmetry Specifications
                sym[0] = (sym[0]+'.')[:-1]
                sym[0] = sym[0][:self.legs[ind][0]]+sym[0][self.legs[ind][-1]+1:]
                sym[1] = sym[1][:self.legs[ind][0]]+sym[1][self.legs[ind][-1]+1:]
                # Create the correct symtensor
                if symlib is None: raise ImportError("Symtensor module not found")
                newten = symlib.SYMtensor(newten,sym=[(self.sym[0]+'.')[:-1],self.sym[1],self.sym[2],self.sym[3]],backend=self.backend)
            else:
                newten = self.ten.array.copy()
                for i in range(len(self.legs[ind])):
                    assert(newten.shape[self.legs[ind][i]+len(sym[0])-1] == 1)
                # Sum over correct legs
                for i in range(len(self.legs[ind]))[::-1]:
                    newten = newten.sum(self.legs[ind][i]+len(sym[0])-1)
                for i in range(len(self.legs[ind]))[::-1]:
                    newten = newten.sum(self.legs[ind][i])
                # Adjust symmetry specifications
                sym[0] = (sym[0]+'.')[:-1]
                sym[0] = sym[0][:self.legs[ind][0]]+sym[0][self.legs[ind][-1]+1:]
                sym[1] = sym[1][:self.legs[ind][0]]+sym[1][self.legs[ind][-1]+1:]
                # Create the correct symtensor
                if symlib is None: raise ImportError("Symtensor module not found")
                newten = symlib.SYMtensor(newten,sym=[(self.sym[0]+'.')[:-1],self.sym[1],self.sym[2],self.sym[3]],backend=self.backend)
        # Update legs
        newlegs = []
        cnt = 0
        for i in range(len(self.legs)):
            if not (i == ind):
                newlegs += [list(range(cnt,cnt+len(self.legs[i])))]
                cnt += len(self.legs[i])
        return GEN_TEN(ten=newten,backend=self.backend,legs=newlegs)

    def merge_inds(self,combinds,make_cp=True):
        """
        Lump multiple indices of a tensor into a single index
        # Note that we can only combine nearest neighbor indices
        # Note also, we are not doing any actual tensor manipulation, i.e. reshaping, etc.
        """
        # Make sure we are only combining nearest neighbor indices
        for i in range(len(combinds)-1):
            assert(combinds[i+1]-combinds[i]==1)
        # Legs that are not affected by merging
        newlegs = []
        for i in range(combinds[0]):
            newlegs.append(self.legs[i])
        # Add the merged legs
        all_comb_inds = []
        for j in range(len(combinds)):
            all_comb_inds += self.legs[combinds[j]]
        newlegs.append(all_comb_inds)
        # Add the rest of the unaffected legs
        for i in range(combinds[-1]+1,len(self.legs)):
            newlegs.append(self.legs[i])
        # Actually change legs
        self.legs = newlegs

    def unmerge_ind(self,ind,make_cp=True):
        """
        Unlump a single index of a tensor into its component indices
        # Note also, we are not doing any actual tensor manipulation, i.e. reshaping, etc.
        """
        # Legs that are not affected by merging
        newlegs = []
        cnt = 0
        for i in range(ind):
            newlegs.append(self.legs[i])
            cnt += len(self.legs[i])
        # Add the unmerged legs
        for i in range(len(self.legs[ind])):
            newlegs.append([cnt])
            cnt += 1
        # Add the rest of the unaffected legs
        for i in range(ind+1,len(self.legs)):
            newlegs.append(list(range(cnt,cnt+len(self.legs[i]))))
            cnt += len(self.legs[i])
        # Actually change legs
        self.legs = newlegs

    def qr(self,split):
        """
        Returns the Q and R from a qr decomposition of the tensor
        """
        #print('\tsplit = {}, Legs = {}, legs[split] = {}'.format(split,self.legs,self.legs[split][0]))
        leg_split = split
        split = self.legs[split][0]
        if self.sym is None:
            # Do qr on non-symmetric tensor
            Q,R = qr_ten(self.ten,split,backend=self.backend)
        else:
            # Do qr on symtensor
            Q,R = symqr(self.ten,[list(range(split)),list(range(split,self.ten.ndim))])
        Q = GEN_TEN(ten=Q,backend=self.backend)
        R = GEN_TEN(ten=R,backend=self.backend)
        # Update Q legs
        Qlegs = []
        cnt = 0
        for i in range(leg_split):
            Qlegs += [list(range(cnt,cnt+len(self.legs[i])))]
            cnt += len(self.legs[i])
        Qlegs += [[cnt]]
        Q.legs = Qlegs
        # Update R legs
        Rlegs = [[0]]
        cnt = 1
        for i in range(leg_split,len(self.legs)):
            Rlegs += [list(range(cnt,cnt+len(self.legs[i])))]
            cnt += len(self.legs[i])
        R.legs = Rlegs
        return Q,R

    def svd(self,split,truncate_mbd=1e100,return_ent=True,return_wgt=True):
        """
        Returns the U,S, and V from an svd of the tensor
        """
        leg_split = split
        split = self.legs[split][0]
        if self.sym is None:
            # Do svd on tensor directly
            res = svd_ten(self.ten,
                          split,
                          backend=self.backend,
                          truncate_mbd=truncate_mbd,
                          return_ent=return_ent,
                          return_wgt=return_wgt)
        else:
            # Do SVD on symmetry blocks
            #tmpprint('\t\t\t\t\t\tGoing to symsvd')
            res = symsvd(self.ten,
                         [list(range(split)),list(range(split,self.ten.ndim))],
                         truncate_mbd=truncate_mbd,
                         return_ent=return_ent,
                         return_wgt=return_wgt)
        #tmpprint('\t\t\t\t\t\tBack from symsvd')
        U,S,V = res[0],res[1],res[2]
        U = GEN_TEN(ten=U,backend=self.backend)
        S = GEN_TEN(ten=S,backend=self.backend)
        V = GEN_TEN(ten=V,backend=self.backend)
        # Update U legs
        Ulegs = []
        cnt = 0
        for i in range(leg_split):
            Ulegs += [list(range(cnt,cnt+len(self.legs[i])))]
            cnt += len(self.legs[i])
        Ulegs += [[cnt]]
        U.legs = Ulegs
        # Update V legs
        Vlegs = [[0]]
        cnt = 1
        for i in range(leg_split,len(self.legs)):
            Vlegs += [list(range(cnt,cnt+len(self.legs[i])))]
            cnt += len(self.legs[i])
        V.legs = Vlegs
        # Bundle results
        ret = (U,S,V)
        for i in range(3,len(res)):
            ret += (res[i],)
        return ret

    def update_signs(self,signs):
        """
        Change the signs of the symtensors
        """
        if self.sym is not None:
            self.sym[0] = signs
            self.ten.sym[0] = signs

    def get_signs(self):
        """
        Change the signs of the symtensors
        """
        if self.sym is not None:
            return self.sym[0]
        else:
            return None

    def flip_signs(self):
        """
        Flip all the signs of a symtensor
        """
        if self.sym is not None:
            self.sym[0] = ''.join(FLIP[i] for i in self.sym[0])
            self.ten.sym[0] = ''.join(FLIP[i] for i in self.ten.sym[0])

    def conj(self):
        return self._as_new_tensor(self.ten.conj())

    def sqrt(self):
        if self.sym is not None:
            return self._as_new_tensor(self.ten**(1./2.))
            #return self._as_new_tensor(self.ten.sqrt())
        else:
            return self._as_new_tensor(self.ten**(1./2.))
            #return self._as_new_tensor(self.backend.sqrt(self.ten))

    def abs(self):
        return self._as_new_tensor(abs(self.ten))

    def sum(self):
        if self.sym is None:
            res = self.backend.einsum('abcdefghijklmnopqrstuvwxyz'[:len(self.ten.shape)]+'->',self.ten)
        else:
            res = self.backend.einsum('abcdefghijklmnopqrstuvwxyz'[:len(self.ten.array.shape)]+'->',self.ten.array)
        return res

    def max(self):
        if self.sym is None:
            maxval = self.backend.max(self.ten)
        else:
            maxval = self.backend.max(self.ten.array)
        return float(maxval)

    def min(self):
        if self.sym is None:
            minval = self.backend.min(self.ten)
        else:
            minval = self.backend.min(self.ten.array)
        return float(minval)

    def to_val(self):
        """
        Returns a single valued tensor's value
        """
        if self.sym is not None:
            tmp = self.ten.array
            es = self.ten.lib.einsum
        else:
            tmp = self.ten
            es = self.lib.einsum
        while True:
            if len(tmp.shape) > 26:
                tmp = es('abcdefghijklmnopqrstuvwxyz...->...',tmp)
            else:
                return es('abcdefghijklmnopqrstuvwxyz'[:len(tmp.shape)]+'->',tmp)

    def __mul__(self,x):
        return self._as_new_tensor(self.ten*x)

    def __rmul__(self,x):
        return self*x

    def __neg__(self):
        return self._as_new_tensor(-self.ten)

    def __div__(self,x):
        return self._as_new_tensor((1./x)*self.ten)

    def __truediv__(self,x):
        return self._as_new_tensor((1./x)*self.ten)

    def __truediv__(self,x):
        return self._as_new_tensor((1./x)*self.ten)

    def __floordiv__(self,x):
        raise NotImplementedError('Floordiv not defined for gen_ten arrays')

    def __rdiv__(self,x):
        return self._as_new_tensor((1./x)*self.ten)

    def __rtruediv__(self,x):
        return self._as_new_tensor((1./x)*self.ten)

    def __rfloordiv__(self,x):
        raise NotImplementedError('Floordiv not defined for gen_ten arrays')

    def __add__(self,x):
        if isinstance(x,GEN_TEN):
            return self._as_new_tensor(self.ten+x.ten)
        else:
            return self._as_new_tensor(self.ten+x)

    def __radd__(self,x):
        if isinstance(x,GEN_TEN):
            return self._as_new_tensor(self.ten+x.ten)
        else:
            return self._as_new_tensor(self.ten+x)

    def __sub__(self,x):
        if isinstance(x,GEN_TEN):
            return self._as_new_tensor(self.ten-x.ten)
        else:
            return self._as_new_tensor(self.tem-x)

    def __setitem__(self, key, value):
        if self.sym:
            self.ten.array[key] = value
        else:
            self.ten[key] = value

    def invert_diag(self):
        newten = self._as_new_tensor(self.ten)
        if newten.sym is None:
            assert(len(self.ten.shape) == 2)
            newten.ten = self.backend.diag(1./self.backend.diag(newten.ten))
        else:
            assert(len(self.ten.array.shape) == 3)
            for i in range(self.ten.array.shape[0]):
                newten.ten.array[i] = self.backend.diag(1./self.backend.diag(newten.ten.array[i]))
        return newten

    def square_inv(self):
        """
        Take the inverse of a 'square' tensor, used in ALS for PEPS Full Update
        """
        newten = self._as_new_tensor(self.ten)
        if newten.sym is None:
            assert(len(self.ten.shape) == 4)
            (n1,n2,n3,n4) = self.ten.shape
            mat = self.backend.reshape(self.ten,(n1*n2,n3*n4))
            inv = self.backend.pinv(mat)
            newten.ten = self.backend.reshape(inv,(n1,n2,n3,n4))
        else:
            # Do the inversion with the full tensor
            mat = self.ten.make_sparse()
            (N1,N2,N3,N4,n1,n2,n3,n4) = mat.shape
            mat = mat.transpose([0,4,1,5,2,6,3,7])
            mat = mat.reshape((N1*n1*N2*n2,N3*n3*N4*n4))
            inv = self.backend.pinv(mat)
            inv = inv.reshape((N1,n1,N2,n2,N3,n3,N4,n4))
            inv = inv.transpose([0,2,4,6,1,3,5,7])
            # Convert back into sparse tensor
            delta = self.ten.get_irrep_map()
            inv = self.backend.einsum('ABCDabcd,ABCD->ABCabcd',inv,delta)
            newten.ten.array = inv
        return newten

    def to_disk(self):

    def from_disk(self):


