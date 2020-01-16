"""
Linear Algebra Tools

Author: Phillip Helms <phelms@caltech.edu>
Date: June 2019

"""

from cyclopeps.tools.params import *
from psutil import virtual_memory as vmem
from shutil import copyfile as _copyfile
import os
import time

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
        tot_mem = bytes2human(vmem()[0])
        av_mem = bytes2human(vmem()[3])
        print('  '*priority+msg+': '+av_mem+' / '+tot_mem)

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
