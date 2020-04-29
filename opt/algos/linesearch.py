import numpy as np
from collections import defaultdict

def backtracking(F, x_k, p, alpha0, rho, c1):
    """
    Backtracking line search algorithm satisfying Armijo condition.
    Reference: Algorithm 3.1 from Nocedal, Numerical Optimization.
    
    Parameters:
    -----------
    F : `function` class
        Objective function.
    x_k : `numpy.ndarry`
        Current iterate.
    p : `numpy.ndarray`
        Descent direction.
    alpha0 : `float`
        Initial step length
    rho : `float`
        In (0,1) backtraking step length reduction factor.
    c1 : `float`
        Constant in sufficient decrease condition 
        f(x_k + alpha_k*p) > f(x_k) + c1*alpha_k*(df_k'*p)
        Typically chosen small, (default 1e-4). 
    
    Returns:
    -------
    alpha : `float`
        Step length.
    info : `dict` of `list`
        Information about the backtracking iteration.    
    """
    # Initialisation 
    alpha = alpha0
    info = defaultdict(list)
    info['alphas'].append(alpha0)
    
    # Loop until condition broken
    while F.f(x_k+alpha*p)>F.f(x_k)+c1*alpha*p.T@F.df(x_k):
        # Calculate new step length
        alpha = rho * alpha
        # Save alpha to list
        info['alphas'].append(alpha)
    
    return alpha, info

def strong_wolfeLS(F, x_k, p_k, alpha_max, c1, c2):
    """
    Line search function satisfying strong Wolfe conditions.
    Reference: Algorithm 3.5 from Nocedal, Numerical Optimization.
    
    It generates a monotonically increasing sequence of step lenghts alpha_j. 
    Uses the fact that interval (alpha_j_1, alpha_j) contains step lengths 
    satisfying strong Wolfe conditions if one of the conditions below is satisfied:
    (C1) alpha_j violates the sufficient decrease condition. 
    (C2) phi(alpha_j) >= phi(alpha_j_1).
    (C3) dphi(alpha_j) >= 0.    

    Parameters:
    -----------
    F : `function` class
        Objective function.
    x_k : `numpy.ndarray`
        Current iterate.
    p_k : `numpy.ndarray`
        Descent direction
    alpha_max : `float`
        Maximum step length.
    c1 : `float`
        Constant in sufficient decrease condition 
        f(x_k + alpha_k*p_k) > f(x_k) + c1*alpha_k*(df_k'*p_k)
        Typically chosen small, (default 1e-4)
    c2 : `float`
        Constant in strong curvature condition 
        |df(x_k + alpha_k*p_k)'*p_k| <= c2*|df(x_k)'*p_k| 
    
    Returns:
    --------
    alpha_s : `float`
        Step length.
    alphas : `list` of `float`
        History of step lengths.
    """
    # Parameters  
    FACT = 10   # Multiple of alpha_j used to generate alpha_{j+1}  
    
    phi  = lambda alpha: F.f(x_k+alpha*p_k) 
    dphi = lambda alpha: F.df(x_k+alpha*p_k).T@p_k    
    
    # Initialisation
    info = defaultdict(list)
    info['alpha'].append(0)
    info['phi'].append(phi(0))
    info['dphi'].append(dphi(0))    
    info['alpha'].append(0.9*alpha_max)
    alpha_s = 0
    n = 1
    max_iter = 10
    stop = False
    
    while n<max_iter and stop == False:
        info['phi'].append(phi(alpha(n)))
        info['dphi'].append(dphi(alpha(n)))      
        if info['phi'][n] > info['phi'][0]+c1*info['alpha'][n]*info['dphi'][0] or (info['phi'][n]>=info['phi'][n-1] and n>1):
            alpha_s, _ = zoom(phi, dphi, info['alpha'][n-1], info['alpha'][n], c1, c2)
            stop = True
        elif np.abs(info['dphi'][n])<=-c2*info['dphi'][0]:
            alpha_s = info['alpha'][n]
            stop = True
        elif info['dphi'][n]>=0:
            alpha_s, _ = zoom(phi, dphi, info['alpha'][n], info['alpha'][n-1], c1, c2)
            stop = True
        info['alpha'].append(min(FACT*info['alpha'][n], alpha_max))
        n += 1
        
    return alpha_s, info['alpha']

def zoom(phi, dphi, alpha_l, alpha_h, c1, c2):
    """
    Zoom algorithm for line search with strong Wolfe conditions.
    Reference: Algorithm 3.6 from Nocedal, Numerical Optimization.
    
    Properties ensured at each iteration:
    (P1) Interval (alpha_l, alpha_h) contains step lengths satisfying strong Wolfe conditions.
    (P2) Among the step lengths generated so far satisfying the sufficient decrease condition
         alpha_l is the one with smallest phi value.
    (P3) alpha_h is chose such that dphi(alpha_l)*(alpha_h - alpha_l) < 0.   
    
    Parameters:
    -----------
    phi : `function` class
        Function of step length phi(alpha) = f(x_k + alpha*p_k) with fields
    dphi : `function` class
        Derivative of function of step length.
    alpha_l : `float`
        Lower boundary of the trial interval.
    alpha_h : `float`
        Upper boundary of the trial interval.
    c1 : `float`
        Constant for Wolfe conditions.
    c2 : `float`
        Constant for Wolfe conditions.
    
    Returns:
    --------
    alpha : `float`
        Step length.
    info : `list` of `float`
        Iteration history.    
    """
    # Parameters
    tol = np.spacing(1)
    n = 0
    stop = False
    max_iter = 100
    
    # Information about the iteration
    info = defaultdict(list)
    
    while n<max_iter and stop == False:
        # Find trial step length alpha_j in [alpha_l, aplha_h]
        alpha_j = 0.5*(alpha_h + alpha_l)
        phi_j   = phi(alpha_j)
        
        # Update info
        info['alpha_ls'].append(alpha_l)
        info['alpha_hs'].append(alpha_h)
        info['alpha_js'].append(alpha_j)
        info['phi_js'].append(phi_j)   
        
        if np.abs(alpha_h-alpha_l)<tol:
            alpha = alpha_j
            stop = True
        
        if phi_j > phi(0) + c1*alpha_j*dphi(0) or phi(alpha_j) >= phi(alpha_l):
            # alpha_j does not satisfy sufficient decrease condition -> look for alpha < alpha_j
            alpha_h = alpha_j
            # Update info
            info['dphi_js'].append(np.nan)
        else:
            # alpha_j satisfies sufficient decrease condition
            dphi_j = dphi(alpha_j)
            # Update info
            info['dphi_js'].append(dphi_j)
            
            if np.abs(dphi_j) <= -c2*dphi(0):
                # alpha_j satisfies strong curvature condition
                alpha = alpha_j
                stop = True
            elif dphi_j*(alpha_h-alpha_l) >= 0:
                # alpha_h : dphi(alpha_l)*(alpha_h - alpha_l) < 0
                # alpha_j violates this condition but swapping alpha_l <-> alpha_h will reestablish it
                # -> [alpha_j, alpha_l]
                alpha_h = alpha_l
            alpha_l = alpha_j
            
    return alpha, info