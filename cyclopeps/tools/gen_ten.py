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
import uuid
from shutil import copyfile as _copyfile

LETTERS = 'abcdefghijklmnoprstuvwxyz'
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
    # Check to make sure tensors are loaded
    if not S.in_mem: raise ValueError('tensor not in memory for calculating entanglement')

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

def eye(D,Z,is_symmetric=False,backend='numpy',dtype=float_,legs=None,in_mem=True):
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
        in_mem : bool
            Whether the tensor should be initially stored in local 
            memory (True) or written to disk (False). Default is
            True.
    """
    if isinstance(backend,str):
        backend = load_lib(backend)
    if not is_symmetric:
        # Create a dense tensor
        ten = backend.eye(D,dtype=dtype)
        ten = GEN_TEN(ten=ten,backend=backend,legs=legs,in_mem=in_mem)
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
        # Write to disk if needed
        if not in_mem: ten.to_disk()
        # Return result
        return ten

def rand(shape,sym=None,backend='numpy',dtype=float_,legs=None,in_mem=True):
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
        in_mem : bool
            Whether the tensor should be initially stored in local 
            memory (True) or written to disk (False). Default is
            True.
    """
    if sym is not None:
        sym = [(sym[0]+',')[:-1],sym[1],sym[2],sym[3]]
    ten = GEN_TEN(shape=shape,
                  sym=sym,
                  backend=backend,
                  dtype=dtype,
                  legs=legs,
                  in_mem=in_mem)
    ten.randomize()
    return ten

def ones(shape,sym=None,backend='numpy',dtype=float_,legs=None,in_mem=True):
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
        in_mem : bool
            Whether the tensor should be initially stored in local 
            memory (True) or written to disk (False). Default is
            True.
    """
    if sym is not None:
        sym = [(sym[0]+',')[:-1],sym[1],sym[2],sym[3]]
    ten = GEN_TEN(shape=shape,
                  sym=sym,
                  backend=backend,
                  dtype=dtype,
                  legs=legs,
                  in_mem=in_mem)
    ten.fill_all(1.)
    return ten

def zeros(shape,sym=None,backend='numpy',dtype=float_,legs=None,in_mem=True):
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
        in_mem : bool
            Whether the tensor should be initially stored in local 
            memory (True) or written to disk (False). Default is
            True.
    """
    if sym is not None:
        sym = [(sym[0]+',')[:-1],sym[1],sym[2],sym[3]]
    ten = GEN_TEN(shape=shape,
                  sym=sym,
                  backend=backend,
                  dtype=dtype,
                  legs=legs,
                  in_mem=in_mem)
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
    # Check to make sure tensors are loaded
    if not opA.in_mem: raise ValueError('operator 1 not in memory for doing einsum')
    if not opB.in_mem: raise ValueError('operator 2 not in memory for doing einsum')

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
    def __init__(self,shape=None,sym=None,backend='numpy',dtype=float_,ten=None,legs=None,
                 writedir=TMPDIR+DIRID,writename=None,in_mem=True):
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
            writedir : str
                The directory where the file will be written to on disk
            writename : str
                The filename where the tensor will be written to on disk
            in_mem : bool
                Whether the tensor should be initially stored in local 
                memory (True) or written to disk (False). Default is
                True.
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

        # Specify the save location and whether the file is saved or loaded
        if writename is None:
            writename = str(uuid.uuid1())
        self.writedir = writedir
        self.saveloc = writedir + '/' + writename
        self.in_mem = True
        if not in_mem:
            self.to_disk()

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
        if self.in_mem:
            return self.ten.shape
        else:
            if self.sym is None:
                return self._saved_shape
            else:
                return self._sym_saved_shape

    @property
    def full_shape(self):
        # Find full shape for tensors stored in memory
        if self.in_mem:

            # Return shape for dense tensor
            if self.sym is None:
                return self.ten.shape

            # Return shape for symtensor
            else:
                shape = list(self.ten.shape)
                for i in range(len(shape)):
                    shape[i] *= len(self.ten.sym[1][i])
                return tuple(shape)

        # Return saved shape for those not in memory
        else:
            return self._saved_shape

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
        # Throw error if tensor is not loaded
        if not self.in_mem: raise ValueError('GEN_TEN not in memory for operation fill_all')

        # Replace current tensor values with random values
        self.ten[:] = 0.
        if self.sym is None:
            self.ten += self.backend.random(self.ten.shape)
        else:
            self.ten += self.backend.random(self.ten.array.shape)

    def fill_all(self,value):
        """
        Fill the tensor with a given value
        """
        # Throw error if tensor is not loaded
        if not self.in_mem: raise ValueError('GEN_TEN not in memory for operation fill_all')

        # Set all tensor values to a single number
        self.ten[:] = 0.
        if self.sym is None:
            self.ten += value*self.backend.ones(self.ten.shape)
        else:
            self.ten += value*self.backend.ones(self.ten.array.shape)

    def make_sparse(self):
        """
        Convert the symmetric tensor into a dense tensor (called
        make_sparse to match symtensor where sparse refers to a dense
        tensor filled with many zeros without symmetry, versus a 
        densely stored symmetric tensor)
        """
        # Throw error if tensor is not loaded
        if not self.in_mem: raise ValueError('GEN_TEN not in memory for operation make_sparse')

        # Return a copy of self if already a full sparse tensor
        if not self.is_symmetric:
            return self.copy()

        # Convert symmetric tensor into a sparse (full dense) version of itself
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
            newten.ten = newten.ten.transpose(order)
            newten.ten = newten.ten.reshape(newshape)
            return newten

    def to_symmetric(self,sym):
        """
        Conver the dense tensor to a symmetric one
        """
        # Throw error if tensor is not loaded
        if not self.in_mem: raise ValueError('GEN_TEN not in memory for operation to_symmetric')

        # Return a copy of self if already a symtensor
        if self.is_symmetric:
            return self.copy()

        # Convert the full dense (sparse in symtensor lang) to symmetric version
        else:
            # Create the new tensor
            newten = self.ten.copy()
            assert(len(sym[0]) == len(newten.shape))
            # Convert the shape
            newshape = []
            for i in range(len(newten.shape)):
                newshape.append(len(sym[1][i]))
                newshape.append(newten.shape[i]/len(sym[1][i]))
            newten = newten.reshape(newshape)
            # Do a transpose on the indices
            order = []
            for i in range(len(sym[1])):
                order.append(2*i)
            for i in range(len(sym[1])):
                order.append(2*i+1)
            newten = newten.transpose(order)
            # Create a random symtensor
            newsymten = rand(newten.shape[len(sym[1]):],
                             sym=sym,
                             backend=self.backend,
                             dtype=self.dtype,
                             legs=self.legs,
                             in_mem=self.in_mem)
            # Contract with delta to get dense irrep
            delta = newsymten.ten.get_irrep_map()
            einstr = LETTERS[:len(sym[1])].upper() + \
                     LETTERS[:len(sym[1])] + ',' + \
                     LETTERS[:len(sym[1])].upper() + '->' + \
                     LETTERS[:len(sym[1])-1].upper() + \
                     LETTERS[:len(sym[1])]
            newten = newsymten.backend.einsum(einstr,newten,delta)
            # Put the result into a symtensor
            newsymten.ten.array = newten
            # Return result
            return newsymten

    def copy(self):
        """
        Return a copy of the gen_ten object
        """
        # Get a copy of a tensor in memory
        if self.in_mem:
            newten = self._as_new_tensor(self.ten.copy())

        # Get a copy of tensors written to disk
        else:

            # Create a new GEN_TEN object with a symtensor array
            if self.sym:
                newten = GEN_TEN(shape = self._sym_saved_shape,
                                 backend = self.backend,
                                 legs = copy.deepcopy(self.legs),
                                 dtype = self.dtype,
                                 writedir = self.writedir,
                                 sym = self.sym,
                                 in_mem = False) # Ensure it is written to disk

            # Create a new GEN_TEN object with dense random array
            else:
                newten = GEN_TEN(shape = self._saved_shape,
                                 backend = self.backend,
                                 legs = copy.deepcopy(self.legs),
                                 dtype = self.dtype,
                                 writedir = self.writedir,
                                 in_mem=False) # Ensure it is written to disk

            # Copy the saved tensors
            try:
                _copyfile(self.saveloc,newten.saveloc)
            except:
                _copyfile(self.saveloc+'.npy',newten.saveloc+'.npy')

            # Update saved values
            newten._max_val = self._max_val
            newten._min_val = self._min_val
            newten._mult_fact = self._mult_fact
            newten._saved_shape = self._saved_shape
            newten._sym_saved_shape = self._sym_saved_shape
            
        # Return result
        return newten

    def _as_new_tensor(self,ten):
        newten = GEN_TEN(ten=ten.copy(),
                         backend=self.backend,
                         legs=copy.deepcopy(self.legs))
        return newten

    def __str__(self):
        # Throw error if tensor is not loaded
        if not self.in_mem: raise ValueError('GEN_TEN not in memory for operation __str__')

        # Return result
        if self.sym:
            return self.ten.array.__str__()
        else:
            return self.ten.__str__()

    def transpose(self,axes):
        """
        Transpose the tensor
        """
        # Throw error if tensor is not loaded
        if not self.in_mem: raise ValueError('GEN_TEN not in memory for operation transpose')
    
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

        # Return new gen ten instance
        return GEN_TEN(ten=newten,backend=self.backend,legs=newlegs)

    def remove_empty_ind(self,ind):
        """
        Remove an index of size 1
        """
        # Throw error if tensor is not loaded
        if not self.in_mem: raise ValueError('GEN_TEN not in memory for operation remove_empty_ind')
    
        # Get initial tensor shape
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

            # Special case when ind is not stored in dense tensor
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
                newten = symlib.SYMtensor(newten,
                                          sym=[(self.sym[0]+'.')[:-1],
                                               self.sym[1],
                                               self.sym[2],
                                               self.sym[3]],
                                          backend=self.backend)

            # General case for remaining tensor legs that are stored
            else:

                # Copy tensor
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
                newten = symlib.SYMtensor(newten,
                                          sym=[(self.sym[0]+'.')[:-1],
                                               self.sym[1],
                                               self.sym[2],
                                               self.sym[3]],
                                          backend=self.backend)
        # Update legs
        newlegs = []
        cnt = 0
        for i in range(len(self.legs)):
            if not (i == ind):
                newlegs += [list(range(cnt,cnt+len(self.legs[i])))]
                cnt += len(self.legs[i])

        # Convert to gen tensor and return
        return GEN_TEN(ten=newten,backend=self.backend,legs=newlegs)

    def merge_inds(self,combinds):
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

    def unmerge_ind(self,ind):
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
        # Throw error if tensor is not loaded
        if not self.in_mem: raise ValueError('GEN_TEN not in memory for operation qr')
    
        # Figure out how legs will be split
        leg_split = split
        split = self.legs[split][0]

        # Do qr on non-symmetric tensor
        if self.sym is None:
            Q,R = qr_ten(self.ten,split,backend=self.backend)

        # Do qr on symtensor
        else:
            Q,R = symqr(self.ten,[list(range(split)),list(range(split,self.ten.ndim))])

        # Convert Q & R back to gen_tens
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

        # Return result
        return Q,R

    def svd(self,split,truncate_mbd=1e100,return_ent=True,return_wgt=True):
        """
        Returns the U,S, and V from an svd of the tensor
        """
        # Throw error if tensor is not loaded
        if not self.in_mem: raise ValueError('GEN_TEN not in memory for operation svd')
    
        # Figure out how legs will be split
        leg_split = split
        split = self.legs[split][0]

        # Do svd on dense tensor directly
        if self.sym is None:
            res = svd_ten(self.ten,
                          split,
                          backend=self.backend,
                          truncate_mbd=truncate_mbd,
                          return_ent=return_ent,
                          return_wgt=return_wgt)

        # Do SVD on symtensor
        else:
            res = symsvd(self.ten,
                         [list(range(split)),list(range(split,self.ten.ndim))],
                         truncate_mbd=truncate_mbd,
                         return_ent=return_ent,
                         return_wgt=return_wgt)

        # Put results into GEN_TENs
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
        # Throw error if tensor is not loaded
        if not self.in_mem: raise ValueError('GEN_TEN not in memory for operation flip_signs')

        # Change signs for symtensor
        if self.sym is not None:
            self.sym[0] = signs
            self.ten.sym[0] = signs

    def get_signs(self):
        """
        Change the signs of the symtensors
        """
        # Throw error if tensor is not loaded
        if not self.in_mem: raise ValueError('GEN_TEN not in memory for operation flip_signs')

        # Return result for symtensor
        if self.sym is not None:
            return self.sym[0]

        # Return None for dense tensor
        else:
            return None

    def flip_signs(self):
        """
        Flip all the signs of a symtensor
        """
        # Throw error if tensor is not loaded
        if not self.in_mem: raise ValueError('GEN_TEN not in memory for operation flip_signs')

        # Do the operation
        if self.sym is not None:
            self.sym[0] = ''.join(FLIP[i] for i in self.sym[0])
            self.ten.sym[0] = ''.join(FLIP[i] for i in self.ten.sym[0])

    def conj(self):
        # Throw error if tensor is not loaded
        if not self.in_mem: raise ValueError('GEN_TEN not in memory for operation conj')

        # Return result as a new tensor
        return self._as_new_tensor(self.ten.conj())

    def sqrt(self):
        # Throw error if tensor is not loaded
        if not self.in_mem: raise ValueError('GEN_TEN not in memory for operation sqrt')

        # Do operation for dense tensor
        if self.sym is not None:
            return self._as_new_tensor(self.ten**(1./2.))

        # Do operation for symtensor
        else:
            return self._as_new_tensor(self.ten**(1./2.))

    def abs(self):
        # Throw error if tensor is not loaded
        if not self.in_mem: raise ValueError('GEN_TEN not in memory for operation abs')

        # Return result as a new tensor
        return self._as_new_tensor(abs(self.ten))

    def sum(self):
        # Throw error if tensor is not loaded
        if not self.in_mem: raise ValueError('GEN_TEN not in memory for operation sum')

        # Do operation for dense tensor
        if self.sym is None:
            einstr = 'abcdefghijklmnopqrstuvwxyz'[:len(self.ten.shape)]+'->'
            res = self.backend.einsum(einstr,self.ten)

        # Do operation for symtensor
        else:
            einstr = 'abcdefghijklmnopqrstuvwxyz'[:len(self.ten.array.shape)]+'->'
            res = self.backend.einsum(einstr,self.ten.array)

        # Return result
        return res

    def max(self):
        # Find result for tensor stored in memory
        if self.in_mem:

            # Find result for dense tensor
            if self.sym is None:
                maxval = self.backend.max(self.ten)

            # Find result for symtensor
            else:
                maxval = self.backend.max(self.ten.array)

        # Find result for tensor not in memory
        else:
            maxval = self._max_val

        # Return the result
        return float(maxval)

    def min(self):
        # Find result for tensor stored in memory
        if self.in_mem:

            # Find result for dense tensor
            if self.sym is None:
                minval = self.backend.min(self.ten)

            # Find result for symtensor
            else:
                minval = self.backend.min(self.ten.array)

        # Find result for tensor not in memory
        else:
            minval = self._min_val

        # Return the result
        return float(minval)

    def max_abs(self):
        """
        Returns the maximum of the absolute value of the tensor
        """
        # Find result for tensor stored in memory
        if self.in_mem:

            # Find result for Dense tensor
            if self.sym is None:
                maxval = max(self.backend.max(self.ten),self.backend.min(self.ten))

            # Find result for symtensor
            else:
                maxval = max(self.backend.max(self.ten.array),self.backend.min(self.ten.array))

        # Find result for tensor not in memory
        else:
            maxval = max(self._max_val, self._min_val, key=abs)

        # Return result
        return float(maxval)

    def to_val(self):
        """
        Returns a single valued tensor's value
        """
        # Throw error if tensor is not loaded
        if not self.in_mem: raise ValueError('GEN_TEN not in memory for operation to_val')

        # Collect the tensor and einsum function
        if self.sym is not None:
            tmp = self.ten.array
            es = self.ten.lib.einsum
        else:
            tmp = self.ten
            es = self.lib.einsum

        # Repeatedly sum over all indices until we return the result
        while True:
            if len(tmp.shape) > 26:
                tmp = es('abcdefghijklmnopqrstuvwxyz...->...',tmp)
            else:
                return es('abcdefghijklmnopqrstuvwxyz'[:len(tmp.shape)]+'->',tmp)

    def __mul__(self,x):
        # Actually multiply tensor if in memory
        if self.in_mem:
            return self._as_new_tensor(self.ten*x)

        # Keep constant factor if not in memory
        else:
            # Copy the gen_ten
            newten = self.copy()

            # Multiply to track the appropriate factor
            newten.update_mult_fact(1./x)
            return newten

    def __rmul__(self,x):
        return self*x

    def __neg__(self):
        return self * (-1.)

    def __div__(self,x):
        return self * (1./x)

    def __truediv__(self,x):
        return self * (1./x)

    def __truediv__(self,x):
        return self * (1./x)

    def __floordiv__(self,x):
        raise NotImplementedError('Floordiv not defined for gen_ten arrays')

    def __rdiv__(self,x):
        return self * (1./x)

    def __rtruediv__(self,x):
        return self * (1./x)

    def update_mult_fact(self,val):
        """
        If tensor is not in memory, then the multiplication
        factor is updated
        """
        # Throw error if tensor is loaded
        if self.in_mem: raise ValueError('Cannot update mult factor for tensor in memory')

        # Update all params
        self._mult_fact *= val
        self._max_val   *= val
        self._min_val   *= val

    def __rfloordiv__(self,x):
        raise NotImplementedError('Floordiv not defined for gen_ten arrays')

    def __add__(self,x):
        # Throw error if tensor is not loaded
        if not self.in_mem: raise ValueError('GEN_TEN not in memory for operation __add__')

        # Write to a new tensor if both tensors are GEN_TENs
        if isinstance(x,GEN_TEN):
            return self._as_new_tensor(self.ten+x.ten)

        # Write to a new tensor if other tensors not GEN_TEN
        else:
            return self._as_new_tensor(self.ten+x)

    def __radd__(self,x):
        # Throw error if tensor is not loaded
        if not self.in_mem: raise ValueError('GEN_TEN not in memory for operation __radd__')

        # Write to a new tensor if both tensors are GEN_TENs
        if isinstance(x,GEN_TEN):
            return self._as_new_tensor(self.ten+x.ten)

        # Write to a new tensor if other tensors not GEN_TEN
        else:
            return self._as_new_tensor(self.ten+x)

    def __sub__(self,x):
        # Throw error if tensor is not loaded
        if not self.in_mem: raise ValueError('GEN_TEN not in memory for operation __sub__')

        # Write to a new tensor if both tensors are GEN_TENs
        if isinstance(x,GEN_TEN):
            return self._as_new_tensor(self.ten-x.ten)

        # Write to a new tensor if other tensors not GEN_TEN
        else:
            return self._as_new_tensor(self.tem-x)

    def __setitem__(self, key, value):
        # Throw error if tensor is not loaded
        if not self.in_mem: raise ValueError('GEN_TEN not in memory for setting item')

        # Set item for symtensor
        if self.sym:
            self.ten.array[key] = value

        # Set item for standard tensor
        else:
            self.ten[key] = value

    def invert_diag(self):
        # Throw error if tensor is not loaded
        if not self.in_mem: raise ValueError('GEN_TEN not in memory for doing invert_diag')
        
        # Copy tensor temporarilly
        newten = self._as_new_tensor(self.ten)

        # Do Dense tensor 
        if newten.sym is None:
            assert(len(self.ten.shape) == 2)
            newten.ten = self.backend.diag(1./self.backend.diag(newten.ten))

        # Do sparse tensor
        else:
            assert(len(self.ten.array.shape) == 3)
            for i in range(self.ten.array.shape[0]):
                newten.ten.array[i] = self.backend.diag(1./self.backend.diag(newten.ten.array[i]))

        # Return result
        return newten

    def square_inv(self):
        """
        Take the inverse of a 'square' tensor, used in ALS for PEPS Full Update
        """
        # Throw error if tensor is not loaded
        if not self.in_mem: raise ValueError('GEN_TEN not in memory for doing square inverse')

        # Copy tensor temporarily
        newten = self._as_new_tensor(self.ten)

        # Do dense tensor inverse
        if newten.sym is None:
            init_shape = self.ten.shape
            nleg = len(init_shape)
            assert(nleg%2==0)
            left_size, right_size = np.prod(init_shape[:nleg/2]), np.prod(init_shape[nleg/2:])
            mat = self.backend.reshape(self.ten,(left_size,right_size))
            inv = self.backend.pinv(mat)
            newten.ten = self.backend.reshape(inv,init_shape)

        # Do sparse tensor inverse
        else:
            # Split the tensor into two
            nleg = len(self.legs)
            assert(nleg%2 == 0)

            # Do the SVD of the tensor
            U,S,V = self.svd(nleg/2,
                             truncate_mbd=None,
                             return_ent=False,
                             return_wgt=False)
            
            # Invert S
            inds = S.backend.find_less(1.-S.ten.array, 1.-1e-15)
            sshape = S.ten.array.shape
            for loc in inds:
                loc = np.unravel_index(loc, sshape)
                S.ten.array[loc] = 1./S.ten.array[loc]

            # Contract to get the inverse
            einstr = LETTERS[:nleg/2+1] + ',' + \
                     LETTERS[nleg/2:nleg/2+2] + '->' + \
                     LETTERS[:nleg/2]+LETTERS[nleg/2+1]
            inv = einsum(einstr,U,S)
            einstr = LETTERS[:nleg/2+1] + ',' + \
                     LETTERS[nleg/2:nleg+1] + '->' + \
                     LETTERS[:nleg/2]+LETTERS[nleg/2+1:nleg+1]
            inv = einsum(einstr,inv,V)
            newten.ten.array = inv.ten.array

        # Return result
        return newten

    def to_disk(self):
        """
        Write the actual gen_ten tensor to disk in location specified
        by gen_ten.saveloc
        """
        # Only need to write if currently in memory
        if self.in_mem:

            # Save some useful information
            self._max_val = self.max()
            self._min_val = self.min()
            self._mult_fact = 1.

            # Actually write to disk
            if self.sym is None:
                self._sym_saved_shape = None
                self._saved_shape = self.ten.shape

                # Write ctf tensor
                if hasattr(self.ten,'write_to_file'):
                    self.ten.write_to_file(self.saveloc)
                    self._is_ctf = True

                # Write Numpy Tensor
                else:
                    self.backend.save(self.saveloc,self.ten)
                    self._is_ctf = False

                # Overwrite tensor with None
                self.ten = None

            else:
                self._sym_saved_shape = self.ten.shape
                self._saved_shape = self.ten.array.shape

                # Write ctf tensor
                if hasattr(self.ten.array,'write_to_file'):
                    self.ten.array.write_to_file(self.saveloc)
                    self._is_ctf = True

                # Write numpy tensor
                else:
                    self.backend.save(self.saveloc,self.ten.array)
                    self._is_ctf = False

                # Overwrite tensor with None
                self.ten.array = None

            # Switch flag to indicate no longer stored in memory
            self.in_mem = False

    def from_disk(self):
        """
        Read the gen_ten tensor from disk, where it has been previously 
        saved in the location specified by gen_ten.saveloc
        """
        # Only need to load if not already in memory
        if not self.in_mem:

            # Load the tensor
            if self.sym is None:

                # Load ctf tensor
                if self._is_ctf:#hasattr(self.ten,'write_to_file'):
                    self.ten = self.backend.zeros(self._saved_shape)
                    self.ten.read_from_file(self.saveloc)

                # Load numpy tensor
                else:
                    self.ten = self.backend.load(self.saveloc + '.npy')

                # Multiply by factor
                self.ten *= self._mult_fact

            else:

                # Load ctf tensor
                if self._is_ctf:#hasattr(self.ten.array,'write_to_file'):
                    self.ten.array = self.backend.zeros(self._saved_shape)
                    self.ten.array.read_from_file(self.saveloc)
                
                # Load Numpy tensor
                else:
                    self.ten.array = self.backend.load(self.saveloc + '.npy')

            # Switch flag to indicate stored in memory
            self.in_mem = True

            # Overwrite the "useful info" saved with tensor
            self._min_val = None
            self._max_val = None
            self._mult_fact = None
            self._saved_shape = None
            self._sym_saved_shape = None
