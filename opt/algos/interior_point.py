import numpy as np
import time
from collections import defaultdict

from opt.algos.descent import feasible_newtonLS
from opt.algos.linesearch import backtracking

def barrier_method(P, q, G, h, A, b, x0, t, mu, tol, max_iter):
    """
    Barrier method 
    
    Parameters:
    -----------
    P : `numpy.ndarray`
        (d,d) 
    q : `numpy.ndarray`
        (d,1)    
    G : `numpy.ndarray`
        (m,d) matrix from inequality constraints.
    h : `numpy.ndarray`
        (m,1)            
    A : `numpy.ndarray`
        (p,d) matrix from equality constraint. 
    b : `numpy.ndarray`
            (p,1)
    x0 : `numpy.ndarray`
        Initial iterate.
    t : `float`
        Parameter for barrier.    
    mu : `float`
        Increase factor for t.
    tol : `float`
        tolerance on the (duality gap ~ m/t)
    max_iter : `int`
        Maximum number of iterations.
    
    Returns:
    --------
    info : `dict`
        Information about optimisation.
        - x:
        - f(x):
        - t: 
        - iterations: 
        - f_iterates: 
        - dulaity_gaps:
        - newton_iterations:
        - iterations:
        - time_taken:
    """
    # Parameters for centering step
    m = G.shape[0] 
    alpha0 = 1
    c1  = 1e-4
    c2  = 0.9
    rho = 0.5
    
    # Parameters to change
    tolNewton = 1e-12
    maxIterNewton = 100    
    
    # Initialisation
    nIter = 0
    stopCond = False
    x_k = x0
    F = FeasibleNewtonCENT(P, q, G, h, A, b, t)
    
    info = defaultdict(list)
    info['x'] = 0
    info['f(x)'] = 0
    info['t'] = 0
    info['mu'] = mu
    info['iterations'] = 0 
    info['iterates'].append(x_k)
    info['f_iterates'].append(F.f(x_k))
    info['newton_iterations'].append(0)
    info['duality_gaps'].append(m/t)
    
    start_time = time.time()
    while stopCond is False and nIter < max_iter:
        print("Barrier Method Iter: ", nIter)
        # Create function handler for centering step 
        # (needs to be redefined at each step because of changing "t")
        F = FeasibleNewtonCENT(P, q, G, h, A, b, t)
        
        # Line search function (needs to be redefined at each step because of changing F) 
        lsFun = lambda F, x_k, p_k, alpha0: backtracking(F, x_k, p_k, alpha0, rho, c1)
        # Centering step
        x_k, f_k, nIterLS, infoLS = feasible_newtonLS(F, lsFun, alpha0, x_k, tolNewton, maxIterNewton)   
        
        # Check stopping condition using duality gap m/t
        if m/t < tol: 
            stopCond = True

        # Increase t
        t *= mu

        # Store info
        info['iterates'].append(x_k)
        info['f_iterates'].append(f_k)
        info['newton_iterations'].append(nIterLS)
        info['duality_gaps'].append(m/t)

        # Increment number of iterations
        nIter += 1
    time_taken = time.time()-start_time
    
    info['x'] = x_k
    info['f(x)'] = F.f(x_k)
    info['t'] = t
    info['iterations'] = nIter
    info['time_taken'] = round(time_taken, 6)
    return info

class FeasibleNewtonCENT:
    def __init__(self, P, q, G, h, A, b, t):
        """
        The objective function we would would like to minimise
        using Feasible Newton method.
        
        Starting with the optimisation problem
        min  0.5x^TPx + q^Tx
        s.t. Gx <= h
             Ax = b  
             
        x in R^{d}
        G in R^{m x d}
        A in R^{p x d}               
        
        We transform it into
        min  tf(x) + phi(x)
        s.t. Ax = b  
        
        where f(x)   = 0.5x^TPx + q^Tx
              phi(x) = -sum log(h - Gx)    
              
        Such that the Newton step is computed 
        |t*f.d2f + phi.d2f    A^T | |delta x_n | = -|t*f.df + phi.df |
        |        A             0  | |   nu_n   |    |       0        |
         
        Arguments:
        ----------
        P : `numpy.ndarray`
            (d,d) 
        q : `numpy.ndarray`
            (d,1)    
        G : `numpy.ndarray`
            (m,d) matrix from inequality constraints.
        h : `numpy.ndarray`
            (m,1)            
        A : `numpy.ndarray`
            (p,d) matrix from equality constraint. 
        b : `numpy.ndarray`
            (p,1)
        t : `float`
            Parameter for barrier.
        """
        self.F   = ObjectiveFunction(P, q)       
        self.phi = BarrierFunction(G, h)         
        self.A = A
        self.t = t
    
    def f(self, x):
        """
        Parameters:
        -----------
        x : `numpy.ndarray`
            (d,1)
            P,
        Returns:
        --------
        f(x) : `float`
            Evaluating CENT objective function.
        """   
        return self.t*self.F.f(x) + self.phi.f(x)
    
    def df(self, x):
        """
        Parameters:
        -----------
        x : `numpy.ndarray`
            (d,1)
            
        Returns:
        --------
        df(x) : `numpy.ndarray`
            (d+p,1) Evaluating first derivative of the CENT objective function.
        """
        p = self.A.shape[0]
        return np.vstack([self.t*self.F.df(x) + self.phi.df(x), np.zeros((p,1))])
    
    def d2f(self, x):
        """
        Parameters:
        -----------
        x : `numpy.ndarray`
            (d,1)
            
        Returns:
        --------
        d2f(x) : `numpy.ndarray`
            (d+p,d+p) Evaluating second derivative of the CENT objective function.
        """
        p = self.A.shape[0]
        return np.vstack([np.hstack([self.t*self.F.d2f(x) + self.phi.d2f(x), self.A.T]),
                          np.hstack([self.A, np.zeros((p,p))])])   

class ObjectiveFunction:
    def __init__(self, P, q):
        """
        Kernelised soft-margin SVM dual 
        form objective function.
        
        min  0.5x^TPx + q^Tx
        s.t. Gx <= h
             Ax = b
        
        x in R^{d}
        G in R^{m x d}
        A in R^{p x d}        
        
        Arguments:
        -----------
        P : `numpy.ndarray`
            (d,d) 
        q : `numpy.ndarray`
            (d,1)             
        """
        self.P = P
        self.q = q
    
    def f(self, x):
        """
        Parameters:
        -----------
        x : `numpy.ndarray`
            (d,1)
            
        Returns:
        --------
        f(x) : `float`
            Evaluating objective function.
        """
        return np.squeeze(0.5*x.T@self.P@x + self.q.T@x)
    
    def df(self, x):
        """
        Parameters:
        -----------
        x : `numpy.ndarray`
            (d,1)
            
        Returns:
        --------
        df(x) : `numpy.ndarray`
            (d,) Evaluating first derivative of the objective function.
        """        
        return self.P@x + self.q
    
    def d2f(self, x):
        """
        Parameters:
        -----------
        x : `numpy.ndarray`
            (d,1)
            
        Returns:
        --------
        d2f(x) : `numpy.ndarray`
            (d,d) Evaluating second derivative of the objective function.
        """            
        return self.P
    
class BarrierFunction:
    def __init__(self, G, h):
        """
        Logarithmic Barrier function.
        
        Starting with the optimisation problem
        min  0.5x^TPx + q^Tx
        s.t. Gx <= h
             Ax = b        

        phi(x) = -sum log[h-Gx]
        
        x in R^{d}
        G in R^{m x d}
        A in R^{p x d}          
        
        Arguments:
        ----------
        G : `numpy.ndarray`
            (m,d) matrix from inequality constraints.
        h : `numpy.ndarray`
            (m,1)
        """
        self.G = G
        self.h = h
    
    def f(self,x):
        """
        Parameters:
        -----------
        x : `numpy.ndarray`
            (d,1)
            
        Returns:
        --------
        f(x) : `float`
            Evaluating logarithmic barrier function.
        """      
        return -np.sum(np.log(self.h - self.G@x + np.spacing(1)))
    
    def df(self, x):
        """
        Parameters:
        -----------
        x : `numpy.ndarray`
            (d,1)
            
        Returns:
        --------
        df(x) : `numpy.ndarray`
            (d,1) Evaluating first derivative of the logarithmic barrier function.
        """        
        tmp = 1 / (self.h - self.G@x + np.spacing(1))
        return self.G.T@tmp
    
    def d2f(self, x):
        """
        Parameters:
        -----------
        x : `numpy.ndarray`
            (d,1)
            
        Returns:
        --------
        d2f(x) : `numpy.ndarray`
            (d,d) Evaluating second derivative of the logarithmic barrier function.
        """    
        tmp = np.squeeze(1 / (self.h - self.G@x + np.spacing(1)))
        return self.G.T@(np.diag(tmp)**2)@self.G   