from cyclopeps.tools.utils import array

# A bunch of operators that are useful in other files
# Annihilate
Sp = array([[0.,1.],
            [0.,0.]])

# Create
Sm = array([[0.,0.],
            [1.,0.]])

# Particle Number
n = array([[0.,0.], 
           [0.,1.]])

# Vacancy
v = array([[1.,0.],
           [0.,0.]])

# Identity
I = array([[1.,0.],
           [0.,1.]])

# Zeros
z = array([[0.,0.],
           [0.,0.]])

# Spin-x
Sx = 1./2.*array([[0.,1.],
                     [1.,0.]])

# Spin-y
Sy = 1./(2.j)*array([[0.,1.],
                        [-1.,0.]])

# Spin-z
Sz = 1./2.*array([[1.,0.],
                     [0.,-1.]])

# Scaled spin operators
X = 2.*Sx
Z = 2.*Sz
Y = -2.j*Sy
