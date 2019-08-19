from numpy import array as nparray

# A bunch of operators that are useful in other files
Sp = nparray([[0.,1.],
            [0.,0.]])

Sm = nparray([[0.,0.],
            [1.,0.]])

n = nparray([[0.,0.], 
           [0.,1.]])

v = nparray([[1.,0.],
           [0.,0.]])

I = nparray([[1.,0.],
           [0.,1.]])

z = nparray([[0.,0.],
           [0.,0.]])

Sx = 1./2.*nparray([[0.,1.],
                     [1.,0.]])

Sy = 1./(2.j)*nparray([[0.,1.],
                        [-1.,0.]])

Sz = 1./2.*nparray([[1.,0.],
                     [0.,-1.]])

X = 2.*Sx
Z = 2.*Sz
Y = -2.j*Sy
