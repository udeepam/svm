import numpy as np
from collections import defaultdict

def feasible_newtonLS(F, ls, alpha0, x0, tol, max_iter):
    """
    Feasible Newton method: Newton method which starts at a feasible point 
    and subsequently enforces the equality constraints on the step maintaining feasibility.
    Reference: Algorithm 10.1 from Boyd, Convex Optimization.
    
    Parameters:
    -----------
    F : `function`
        Objective function.
    ls : `function`
        Specifies line search algorithm.
    alpha0 : `float`
        initial step length.
    x0 : `numpy.ndarray`
        Initial iterate.
    tol : `float`
        Stopping condition on minimal allowed step.
    max_iter : `int`
        Maximum number of iterations.
    
    Returns:
    --------
    x_min : `numpy.ndarray`
        Minimum of objective function F
    f_min : `float`
        Value of minimum objective function F
    nIter : `int`
        Number of iterations.
    info : `dict` of `list`
        Information about iteration.       
    """
    pass

def descentLS(F, descent, ls, alpha0, x0, tol, max_iter, stop_type):
    """
    Wrapper function executing  descent with line search.
    Descent Methods:
    1. Steepest descent direction
    2. Newton direction
    
    Parameters:
    -----------
    F : `function`
        Objective function.
    descent : `str`
        Specifies descent direction {'steepest', 'newton'}
    ls : `function`
        Specifies line search algorithm.
    alpha0 : `float`
        initial step length.
    x0 : `numpy.ndarray`
        Initial iterate.
    tol : `float`
        Stopping condition on minimal allowed step.
    max_iter : `int`
        Maximum number of iterations.
    stop_type : `str`
        The stopping condition {'step','grad'}
    
    Returns:
    --------
    x_min : `numpy.ndarray`
        Minimum of objective function F
    f_min : `float`
        Value of minimum objective function F
    nIter : `int`
        Number of iterations.
    info : `dict` of `list`
        Information about iteration.    
    """    
    # Initialisation
    nIter = 0
    x_k = x0
    stopCond = False
    
    info = defaultdict(list)
    info['xs'].append(x0)
    info['alphas'].append(alpha0)   
    
    # Loop until convergence or maximum number of iterations
    while stopCond is False:
        # Increment iterations
        nIter += 1
        
        # Compute descent direction
        if descent == "steepest":
            p_k = -F.df(x_k) # steepest descent direction
        elif descent == "newton":
            p_k = -np.linalg.pinv(F.d2f(x_k))@F.df(x_k) # Newton direction
            if p_k.T@F.df(x_k)>0:
                # Force to be descent direction (only active if F.d2f(x_k) not pos.def.)
                p_k = -p_k
                
        # Call line search given by handle ls for computing step length
        alpha_k = ls(x_k, p_k, alpha0)
        
        # Update x_k and f_k
        x_k_1 = x_k
        x_k = x_k + alpha_k*p_k
        
        # Store iteration info
        info['xs'].append(x_k)
        info['alphas'].append(alpha_k)
        
        if stop_type == 'step':
            # Compute relative step length
            norm_step = np.linalg.norm(x_k - x_k_1) / np.linalg.norm(x_k_1)
            stopCond = (normStep < tol)
        elif stop_type == 'grad':
            stopCond = (np.linalg.norm(F.df(x_k)) <= tol)
            
    return x_k, F.f(x_k), nIter, info