from cyclopeps.tools.params import *
from cyclopeps.tools.utils import *
from cyclopeps.tools.peps_tools import *

def run_peps():
    Nx = 10
    Ny = 10
    d = 2
    D = 4
    chi = 10
    peps = PEPS(Nx=Nx,Ny=Ny,d=d,D=D,chi=chi)

if __name__ == "__main__":
    run_peps()
