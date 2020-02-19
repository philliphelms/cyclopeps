"""
A class to hold common basic operators
"""
from symtensor.settings import load_lib
from cyclopeps.tools.gen_ten import zeros

class OPS:
    """
    A class to hold common basic operators

    Included ops are:
        Sp: [[0,1]
             [0,0]]
        Sm: [[0,0]
             [1,0]]
        n:  [[0,0]
             [0,1]]
        v:  [[1,0]
             [0,0]]
        I:  [[1,0]
             [0,1]]
        Sx: 1/2*[[0,1]
                 [1,0]]
        Sy: 1/2j*[[ 0,1]
                  [-1,0]]
        Sz: 1/2*[[1, 0]
                 [0,-1]]
        X:  [[0,1]
             [1,0]]
        Y:  [[0,-1]
             [1, 0]]
        Z:  [[1, 0]
             [0,-1]]

    Kwargs:
        sym : str
            The type of symmetry for the operators.
            Possibilities are currently:
                [None, 'Z2']
        backend : str
            The backend to be used in generating the 
            tensors, i.e. 'ctf' or 'numpy'
    """
    def __init__(self,sym=None,backend='numpy'):
        """
        Create the OPS object
        """
        # Get the tensor backend
        if isinstance(backend,str):
            self.backend = load_lib(backend)
        else:
            self.backend = backend

        # Save symmetry type
        self.sym = sym

        # Create all operators
        if sym is None:
            self.load_ops()
        elif sym == 'Z2':
            self.load_z2_ops()

    def load_ops(self):
        # Annihilate
        Sp = zeros((2,2),sym=self.sym,backend=self.backend)
        Sp[0,1] = 1.

        # Create
        Sm = zeros((2,2),sym=self.sym,backend=self.backend)
        Sm[1,0] = 1.

        # Particle Number
        n = zeros((2,2),sym=self.sym,backend=self.backend)
        n[1,1] = 1.

        # Vacancy
        v = zeros((2,2),sym=self.sym,backend=self.backend)
        v[0,0] = 1.

        # Identity
        I = zeros((2,2),sym=self.sym,backend=self.backend)
        I[0,0] = 1.
        I[1,1] = 1.

        # Zeros
        z = zeros((2,2),sym=self.sym,backend=self.backend)

        # Spin-x
        Sx = zeros((2,2),sym=self.sym,backend=self.backend)
        Sx[0,1] = 1./2.
        Sx[1,0] = 1./2.

        # Spin-y
        #Sy = zeros((2,2),sym=self.sym,backend=self.backend)
        #Sy[0,1] = 1./(2.j)
        #Sy[1,0] = -1./(2.j)

        # Spin-z
        Sz = zeros((2,2),sym=self.sym,backend=self.backend)
        Sz[0,0] = 1./2.
        Sz[1,1] = -1./2.

        # Scaled Spin X
        X = zeros((2,2),sym=self.sym,backend=self.backend)
        X[0,1] = 1.
        X[1,0] = 1.

        # Scaled Spin Z
        Z = zeros((2,2),sym=self.sym,backend=self.backend)
        Z[0,0] = 1.
        Z[1,1] = -1.

        # Scaled Spin Y

        # Save results
        self.Sp = Sp
        self.Sm = Sm
        self.n = n
        self.v = v
        self.I = I
        self.z = z
        self.Sx = Sx
        self.Sz = Sz
        self.X = X
        self.Z = Z

    def load_z2_ops(self):
        raise NotImplementedError()
