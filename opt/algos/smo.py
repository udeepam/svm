import numpy as np
import time
import copy
from collections import defaultdict

def smo(y, K, C, loss_fn, tol=1e-3, max_iter=5, **kwargs):
    """
    Sequential Minimal Optimisation (SMO)
    References: CS 229 notes
    
    Parameters:
    -----------
    y : `numpy.ndarray`
        (nData,1) matrix of corresponding labels. Each element is in the set {-1,+1}.
    K : `numpy.ndarray`
        (nData, nData) Kernel matrix of training data.    
    C : `float`
        Regularization parameter. The strength of the regularization is 
        inversely proportional to C. Must be strictly positive. 
        The penalty is a squared l2 penalty.
    loss_fn : `str`
        The loss function for the optimisation problem. {'L1','L2'}.           
    tol : `float`
        Tolerance.
    max_iter : `int`
        Maximum number of iterations.
    
    Returns:
    --------
    info : `dict`
        Information about optimisation.
        - x:
        - f(x):
        - iterations: 
        - f_iterates: 
        - iterations:
        - time_taken:
    """    
    n, _ = y.shape
    nIter = 0
    
    # Initialisation
    lambdas = np.zeros((n,1), dtype=float)    
    w0 = 0
    E  = np.zeros((n,1), dtype=float)
    info = defaultdict(list)
    info['iterates'].append(copy.deepcopy(lambdas))
    
    start_time = time.time()
    while nIter < max_iter:
        print("SMO method iteration: ", nIter)
        num_changed_lambdas = 0
        for i in range(n):        
            # Calculate E_i
            E[i] = w0 + (lambdas*y).T@K[:,i] - y[i]
            if (y[i]*E[i] < -tol and lambdas[i] < C) or (y[i]*E[i] > tol and lambdas[i] > 0):
                j = np.random.randint(0, n)
                # Make sure j does not equal i
                while j == i:
                    j = np.random.randint(0, n)
                
                # Calculate E_j
                E[j] = w0 + (lambdas*y).T@K[:,j] - y[j]
                
                # Save old lambdas
                lambda_i_old = lambdas[i]
                lambda_j_old = lambdas[j]
                    
                # Compute eta
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue  
                
                # Compute lambda_j 
                lambdas[j] = lambdas[j] - (y[j]*(E[i] - E[j]))/eta
                
                # Compute L and H and clip
                if loss_fn == "L1":
                    if y[i] != y[j]:
                        L = max(0, lambdas[j] - lambdas[i])
                        H = min(C, C + lambdas[j] - lambdas[i])                    
                    else:
                        L = max(0, lambdas[i] + lambdas[j] - C)
                        H = min(C, lambdas[i] + lambdas[j])
                    if L == H:
                        # Store iterates
                        info['iterates'].append(copy.deepcopy(lambdas))                            
                        continue
                    # Clip
                    lambdas[j] = min(H, lambdas[j])
                    lambdas[j] = max(L, lambdas[j])                         
                elif loss_fn == "L2":
                    lambdas[j] = max(0, lambdas[j])  
                
                # Check if change in labda is significant
                if np.abs(lambdas[j] - lambda_j_old) < 1e-5:
                    # Store iterates
                    info['iterates'].append(copy.deepcopy(lambdas))                        
                    continue

                # Determine value for lambda_i
                lambdas[i] = lambdas[i] + y[i]*y[j]*(lambda_j_old - lambdas[j])

                # Compute b1 and b2
                b1 = w0 - E[i] - y[i]*(lambdas[i] - lambda_i_old)*K[i, i] - y[j]*(lambdas[j] - lambda_j_old)*K[i, j]
                b2 = w0 - E[j] - y[i]*(lambdas[i] - lambda_i_old)*K[i, j] - y[j]*(lambdas[j] - lambda_j_old)*K[j, j]
                
                # Compute w0
                if loss_fn == "L1":
                    if 0 < lambdas[i] and lambdas[i] < C:
                        w0 = b1
                    elif 0 < lambdas[j] and lambdas[j] < C:
                        w0 = b2
                    else:
                        w0 = (b1 + b2) / 2
                elif loss_fn == "L2":
                    if 0 < lambdas[i]:
                        w0 = b1
                    elif 0 < lambdas[j]:
                        w0 = b2
                    else:
                        w0 = (b1 + b2) / 2                                         
                num_changed_lambdas += 1
                # Store iterates
                info['iterates'].append(copy.deepcopy(lambdas))
        if num_changed_lambdas == 0:
            nIter += 1
        else:
            nIter = 0 
            
    time_taken = time.time()-start_time
    # Save optimisation info
    info['x'] = lambdas
    info['w0'] = w0
    info['time_taken'] = round(time_taken, 6)    
    return info