import numpy as np

def feasible_starting_point(y, C):
    """
    Calculate feasible starting point for 
    Barrier method specific for SVM dual problem.
      
    Parameters:
    -----------
    y : `numpy.ndarray`
        (nData,) The labels.
    C : `float`
        Regularization parameter. The strength of the regularization is 
        inversely proportional to C. Must be strictly positive. 
        The penalty is a squared l2 penalty.    
    
    Returns:
    --------
    x0 : numpy.ndarray`
        (nData,1) The starting point for the feasible Newton method.
    """
    nData = len(y)
    nPos = np.sum(y==1)
    pos_neg_frac = nPos / (nData-nPos)
    x0 = np.zeros((nData,1))
    for i in range(nData):
        if y[i]==1:
            x0[i] = C*(1-nPos/nData)
        else:
            x0[i] = C*pos_neg_frac*(1-nPos/nData)
    return x0