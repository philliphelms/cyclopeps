"""
A wrapper for generic and symmetric tensors
providing required functionality for PEPS calculations

Author: Phillip Helms <phelms@caltech.edu>
Date: January 2020

"""
from numpy import float_
import symtensor.sym as symlib
from symtensor.settings import load_lib
import itertools
import sys

###########################################################
# Functions
###########################################################
def eye(D,Z,is_symmetric=False,backend='numpy',dtype=float_):
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
        ten = GEN_TEN(ten=ten,backend=backend)
        return ten
    else:
        # Create a symmetric tensor
        sym = ['+-',[Z,Z],None,None]
        ten = GEN_TEN(shape=(D,D),sym=sym,backend=backend,dtype=dtype)
        for i in range(ten.ten.array.shape[0]):
            ten.ten.array[i,:,:] = backend.eye(ten.ten.array.shape[1])
        return ten

def rand(shape,sym=None,backend='numpy',dtype=float_):
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
    ten = GEN_TEN(shape=shape,sym=sym,backend=backend,dtype=dtype)
    ten.randomize()
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
    # Format String ------------------------------------
    _subscripts = subscripts
    subscripts = replace_caps(subscripts)
    subscripts = unmerge_subscripts(subscripts,opA.legs,opB.legs)
    print('{}'.format(subscripts))
    print('{},{}->'.format(opA.sym[0],opB.sym[0]))
    #print(opA.shape,opA.legs)
    #print(opB.shape,opB.legs)
    # Do einsum
    res = opA.lib.einsum(subscripts,opA.ten,opB.ten)
    # Create a new gen_ten (with correctly lumped legs) from the result
    # Find resulting sym
    if opA.sym is not None:
        sym = res.sym
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
        else:
            legs += [list(range(cnt,cnt+len(opB.legs[strB_loc])))]
        cnt += len(opA.legs[strA_loc])
    res = GEN_TEN(sym = sym,
                  backend = opA.backend,
                  ten = res,
                  legs = legs)
    print('{},{}->{}'.format(opA.sym[0],opB.sym[0],res.sym[0]))
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
        # Collect settings
        if isinstance(backend,str):
            self.backend = load_lib(backend)
        else:
            self.backend = backend
        if ten is None:
            self.dtype   = dtype
        else:
            self.dtype   = ten.dtype
        self.sym     = sym

        if ten is None:
            # Create a zero tensor
            if sym is None:
                self.ten = self.lib.zeros(shape,dtype=dtype)
            else:
                self.ten = symlib.zeros(shape,sym=sym,backend=backend,dtype=dtype)
        else:
            self.ten = ten
        
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
        if self.sym is None:
            self.ten += self.backend.random(self.ten.shape)
        else:
            self.ten += self.backend.random(self.ten.array.shape)

    def copy(self):
        """
        Return a copy of the gen_ten object
        """
        return self._as_new_tensor(self.ten)

    def _as_new_tensor(self,ten):
        newten = GEN_TEN(self.shape,ten=ten,sym=self.sym,backend=self.backend,dtype=self.dtype,legs=self.legs)
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
        self.ten.transpose(*_axes)
        # Update legs
        newlegs = []
        ind = 0
        for i in range(len(axes)):
            newlegs.append(list(range(ind,ind+len(self.legs[axes[i]]))))
            ind += len(self.legs[axes[i]])
        self.legs = newlegs

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

    def conj(self):
        return self._as_new_tensor(self.ten.conj())

    def to_val(self):
        """
        Returns a singular value
        """
        if self.sym is not None:
            return self.ten.lib.einsum('abcdefghijklmnopqrstuvwxyz'[:len(self.ten.array.shape)]+'->',self.ten.array)
        else:
            return self.lib.einsum('abcdefghijklmnopqrstuvwxyz'[:len(self.ten.shape)]+'->',self.ten)

    def __mul__(self,x):
        return self._as_new_tensor(self.ten*x)

    def __neg__(self,x):
        return self._as_new_tensor(-self.ten)

    def __div__(self,x):
        return self._as_new_tensor(self.ten/x)

    def __add__(self,x):
        if isinstance(x,GEN_TEN):
            return self._as_new_tensor(self.ten+x.ten)
        else:
            return self._as_new_tensor(self.ten+x)

    def __sub__(self,x):
        if isinstance(x,GEN_TEN):
            return self._as_new_tensor(self.ten-x.ten)
        else:
            return self._as_new_tensor(self.tem-x)
