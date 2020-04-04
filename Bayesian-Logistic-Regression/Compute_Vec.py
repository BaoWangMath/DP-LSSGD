# -*- coding: utf-8 -*-
"""
Compute the vector of the first row of A^{-1/2}
"""
import numpy as np

def Compute_Vec(len_param, sigma):
    if len_param > 2:
        w1 = np.exp(1j*2*np.pi/len_param) # Unit root of 1
        #vecs_len.append(len_param)
        Mat = np.zeros((len_param, len_param))
        Mat[0, 0] = 1.+2.*sigma; Mat[0, 1] = -sigma; Mat[0, -1] = -sigma
        Mat[-1, 0] = -sigma; Mat[-1, -2] = -sigma; Mat[-1, -1] = 1.+2*sigma
        for i in range(1, len_param-1):
            Mat[i, i-1] = -sigma; Mat[i, i] = 1+2*sigma; Mat[i, i+1] = -sigma
        
        # Assemble eigenvectors into a matrix with ith row represent the ith eigenvector
        A = np.zeros((len_param, len_param), dtype='complex128')
        for i in range(len_param):
            wi = w1**i
            for j in range(0, len_param):
                A[i, j] = wi**j/np.sqrt(len_param)
        A = A.T
        
        # Eigenvectors
        c = np.zeros((len_param, 1), dtype='complex128')
        c[0] = 1.+2.*sigma
        c[1] = -1.*sigma
        c[-1] = -1.*sigma
        
        eigvals = np.zeros((len_param, 1))
        for i in range(0, len_param):
            wi = w1**i
            lambdai = c[0] + c[1]*wi**(len_param-1) + c[-1]*wi**1
            eigvals[i] = lambdai
        
        # Compute the square root of the origial matrix
        EigMat = np.zeros((len_param, len_param))
        for i in range(len_param):
            EigMat[i, i] = np.sqrt(eigvals[i])
        
        Sqrt_Mat = np.matmul(np.matmul(A, EigMat), A.T)
        conv_vec = (Sqrt_Mat[0, :].real).reshape((1, len_param))
        return conv_vec