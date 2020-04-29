import numpy as np
import numexpr as ne
from scipy.linalg.blas import dgemm

def polynomial_kernel_matrix(P, Q, c, degree):
    """
    Calculate kernel matrix using polynomial kernel.
    
    k(p,q) = (p^{T}q + c)^d

    Parameters:
    -----------
    P : `numpy.ndarray`
        (nDataP, nDim) matrix of data. Each row corresponds to a data point.
    Q : `numpy.ndarray`
        (nDataQ, nDim) matrix of data. Each row corresponds to a data point.   
    c : `float`
        Bias term, c >= 0.      
    degree : `int`
        Degree of the polynomial kernel function.

    Returns:
    --------
    K : `numpy.ndarray`
        (nDataP,nDataQ) matrix, the polynomial kernel matrix of the P and Q data matrix.        
    """
    P = P.astype('float32')
    Q = Q.astype('float32')
    return ne.evaluate('(c + A)**d', {
        'A' : dgemm(alpha=1.0, a=P, b=Q, trans_b=True),
        'c' : c,
        'd' : degree
    })       


def gaussian_kernel_matrix(P, Q, c):
    """
    Calculate kernel matrix using gaussian kernel.
    
    ||p-q||^2 = ||p||^2 + ||q||^2 - 2 * p^T * q

    k(p,q) = exp(-c*||p-q||^2)
           = exp(-c*[||p||^2 + ||q||^2 - 2 * p^T * q])

    Parameters:
    -----------
    P : `numpy.ndarray`
        (nDataP, nDim) matrix of data. Each row corresponds to a data point.
    Q : `numpy.ndarray`
        (nDataQ, nDim) matrix of data. Each row corresponds to a data point.            
    c : `int`
        Width of the gaussian kernel function.

    Returns:
    --------
    K : `numpy.ndarray`
        (nDataP,nDataQ) matrix, the gaussian kernel matrix of the P and Q data matrix.                 
    """
    # Calculate norm
    P_norm = np.einsum('ij,ij->i',P,P,dtype='float32')
    Q_norm = np.einsum('ij,ij->i',Q,Q,dtype='float32')
    return ne.evaluate('exp(-gamma * (A + B - 2*C))', {
        'A' : P_norm[:,None],
        'B' : Q_norm[None,:],
        'C' : dgemm(alpha=1.0, a=P, b=Q, trans_b=True),
        'gamma' : c
    })