"""
Linear Algebra Tools

Author: Phillip Helms <phelms@caltech.edu>
Date: June 2019

"""

from cyclopeps.tools.params import *
try:
    from psutil import virtual_memory as vmem
except:
    pass
from shutil import copyfile as _copyfile
import os
import time
try:
    import h5py
except:
    h5py = None
import sys

def bytes2human(n):
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n) / prefix[s]
            return '%.1f%s' % (value, s)
    return "%sB" % n


def mpiprint(priority,msg):
    """ 
    Make print only work when RANK == 0, 
    avoiding repeated printing of statements.
    """
    if (RANK == 0) and (priority <= VERBOSE_MEM):
        tot_mem = bytes2human(vmem()[0])
        av_mem = bytes2human(vmem()[3])
        try:
            msg += ': '+av_mem+' / '+tot_mem
        except:
            pass
    if (RANK == 0) and (priority <= VERBOSE):
        try:
            print('  '*priority+msg)
        except:
            print(msg)
    sys.stdout.flush()

# Create a print function for when print statements are temporary
def tmpprint(msg):
    """
    Print function for when we are printing something only temporarily
    i.e. use this function if you want to be able to print something to debug,
    then search for all instances of tmpprint to remove extra print statements
    """
    mpiprint(0,msg)

def timeprint(priority,msg):
    """ 
    Make print only work when RANK == 0, 
    avoiding repeated printing of statements.
    """
    if (RANK == 0) and (priority <= VERBOSE_TIME):
        try:
            print('  '*priority+msg+' '+str(time.time()))
        except:
            print(msg+' '+str(time.time()))

def memprint(priority,msg):
    """ 
    Make print only work when RANK == 0, 
    avoiding repeated printing of statements.
    """
    if (RANK == 0) and (priority <= VERBOSE_MEM):
        try:
            tot_mem = bytes2human(vmem()[0])
            av_mem = bytes2human(vmem()[3])
            print('  '*priority+msg+': '+av_mem+' / '+tot_mem)
        except: 
            pass

def mkdir(path):
    """
    Wrapper for making a directory
    """
    if (RANK == 0):
        try:
            os.mkdir(path)
        except:
            pass

def copyfile(old_fname,new_fname):
    """
    Wrapper to copy a file
    """
    if (RANK == 0):
        try:
            _copyfile(old_fname,new_fname)
        except:
            _copyfile(old_fname+'.npy',new_fname+'.npy')

def load_lib(libname):
    if isinstance(libname,str):
        if libname == 'numpy':
            try:
                import symtensor.backend.numpy_funclib as lib
            except:
                import cyclopeps.tools.numpy_funclib as lib
        elif libname == 'ctf':
            try:
                import symtensor.backend.ctf_funclib as lib
            except Exception as e:
                import cyclopeps.tools.ctf_funclib as lib
        elif libname == 'blas':
            try:
                import symtensor.backend.blas_funclib as lib 
            except:
                import cyclopeps.tools.blas_funclib as lib
        else:
            raise ValueError("Library %s not recognized" %libname)
        return lib
    else:
        return libname

def open_file(fname,rw):
    if h5py is not None:
        return h5py.File(fname,rw)
    else: 
        return None

def create_dataset(f,data_label,data):
    if h5py is not None:
        f.create_dataset(data_label,data=data)

def get_dataset(f,data_label):
    if h5py is not None:
        return f[data_label][...]
    else:
        return None

def close_file(f):
    if h5py is not None:
        f.close()
