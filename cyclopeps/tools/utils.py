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
import h5py
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
            except:
                import cyclopeps.tools.ctf_funclib as lib
        else:
            raise ValueError("Library %s not recognized" %libname)
        return lib
    else:
        return libname

def open_file(fname,rw):
    return h5py.File(fname,rw)

def create_dataset(f,data_label,data):
    f.create_dataset(data_label,data=data)

def get_dataset(f,data_label):
    return f[data_label][...]

def close_file(f):
    f.close()
