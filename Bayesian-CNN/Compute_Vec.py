# -*- coding: utf-8 -*-
"""
Compute the vector of the first row of A^{-1/2}
"""
import numpy as np

def Compute_Vec(len_param, sigma):
    # First row of the eigen matrix
    A_first_row = np.ones((len_param,), dtype='complex128')/np.sqrt(len_param)
    
    # First row of A
    c = np.zeros((len_param,), dtype='complex128')
    c[0] = 1.+2.*sigma
    c[1] = -1.*sigma
    c[-1] = -1.*sigma
    
    # Eigenvalues of A^{-1/2}
    eigvals = np.zeros((len_param,), dtype='float32')
    w1 = np.exp(1j*2*np.pi/len_param) # Unit root of 1
    for i in range(0, len_param):
        wi = w1**i
        eigvals[i] = np.sqrt(c[0]+c[1]*wi**(len_param-1) + c[-1]*wi**1)
    
    vec1 = A_first_row
    for i in range(len(vec1)):
        vec1[i] = vec1[i]*eigvals[i]
    
    # Compute first row of A^{-1/2}
    conv_vec = np.zeros((len_param,), dtype='float32')
    for i in range(0, len_param):
        # i-th column of A^T
        vec_tmp = np.zeros((len_param,), dtype='complex128')
        for ii in range(0, len_param):
            vec_tmp[ii] = w1**(i*ii)
        conv_vec[i] = np.real(np.dot(vec1, vec_tmp))/np.sqrt(len_param)
    
    return conv_vec

if __name__ == '__main__':
    print 'The vector is: ', Compute_Vec(5, 1)