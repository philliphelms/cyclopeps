from cyclopeps.tools.utils import array

# A bunch of operators that are useful in other files
Sp = array([[0.,1.],
            [0.,0.]])

Sm = array([[0.,0.],
            [1.,0.]])

n = array([[0.,0.], 
           [0.,1.]])

v = array([[1.,0.],
           [0.,0.]])

I = array([[1.,0.],
           [0.,1.]])

z = array([[0.,0.],
           [0.,0.]])

Sx = 1./2.*array([[0.,1.],
                     [1.,0.]])

Sy = 1./(2.j)*array([[0.,1.],
                        [-1.,0.]])

Sz = 1./2.*array([[1.,0.],
                     [0.,-1.]])

X = 2.*Sx
Z = 2.*Sz
Y = -2.j*Sy
