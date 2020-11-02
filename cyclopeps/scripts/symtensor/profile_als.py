from cyclopeps.tools.utils import *
from cyclopeps.algs.tebd import alternating_least_squares
from sys import argv
import cProfile
from memory_profiler import profile 
from cyclopeps.tools.gen_ten import rand

# Get inputs
D  = int(argv[1])
Zn = int(argv[2])
if Zn == 0:
    Zn = None
backend = argv[3]
d = 2

if backend == 'ctf':
    import ctf
    wrld = ctf.comm()

# Create needed tensors
if Zn is None:
    phys_b = rand((d*D,d,D),
                  None,
                  backend=backend,
                  dtype=float)
    phys_t = rand((D,d,d*D),
                  None,
                  backend=backend,
                  dtype=float)
    N =      rand((d*D,d*D,d*D,d*D),
                  None,
                  backend=backend,
                  dtype=float)
    eH =     rand((d,d,d,d),
                  None,
                  backend=backend,
                  dtype=float)
else:
    phys_b = rand((D,1,D/Zn),
                  ['++-',[range(Zn)]*3,0,Zn],
                  backend=backend,
                  dtype=float)
    phys_t = rand((D/Zn,1,D),
                  ['++-',[range(Zn)]*3,0,Zn],
                  backend=backend,
                  dtype=float)
    N =      rand((D,D,D,D),
                  ['+--+',[range(Zn)]*4,0,Zn],
                  backend=backend,
                  dtype=float)
    eH =     rand((1,1,1,1),
                  ['--++',[range(Zn)]*4,0,Zn],
                  backend=backend,
                  dtype=float)

# Run the calculation
if backend == 'ctf': 
    from ctf import timer_epoch
    te = timer_epoch('1')
    te.begin()
t0 = time.time()
profile_fname = 'run_als_stats_d{}_D{}_Zn{}_{}'.format(d,D,Zn,backend)
cProfile.run('out = alternating_least_squares(phys_b,phys_t,N,eH,D,als_iter=1,als_tol=1e-16)',profile_fname)
tf = time.time()
print('ALS Time: {}'.format(tf-t0))

# End ctf
if backend == 'ctf': 
    te.end()
