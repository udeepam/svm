import numpy as np


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
        return np.vstack([self.t*self.F.df(x) + self.phi.df(x), np.zeros(p,1)])
    
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
        print(self.h - self.G@x)
        return -np.sum(np.log(self.h - self.G@x))
    
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
        tmp = 1 / (self.h - self.G@x)
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
        tmp = np.squeeze(1 / (self.h - self.G@x))
        return self.G.T@(np.diag(tmp)**2)@self.G  

# class CENT:
#     def __init__(self, P, q, G, h, A, b, t):
#         """
#         The objective function we would would like to minimise.
        
#         Starting with the optimisation problem
#         min  0.5x^TPx + q^Tx
#         s.t. Gx <= h
#              Ax = b  
             
#         x in R^{d}
#         G in R^{m x d}
#         A in R^{p x d}               
        
#         We transform it into
#         min  tf(x) + phi(x)
#         s.t. Ax = b  
        
#         where f(x)   = 0.5x^TPx + q^Tx
#               phi(x) = -sum log(h - Gx)             
        
#         Arguments:
#         ----------
#         P : `numpy.ndarray`
#             (d,d) 
#         q : `numpy.ndarray`
#             (d,1)    
#         G : `numpy.ndarray`
#             (m,d) matrix from inequality constraints.
#         h : `numpy.ndarray`
#             (m,1)            
#         A : `numpy.ndarray`
#             (p,d) matrix from equality constraint. 
#         b : `numpy.ndarray`
#             (p,1)
#         t : `float`
#             Parameter for barrier.
#         """
#         self.P = p
#         self.P = q
#         self.G = G
#         self.h = h        
#         self.A = A
#         self.b = b        
#         self.t = t
    
#     def f(self, x):
#         """
#         Parameters:
#         -----------
#         x : `numpy.ndarray`
#             (d,1)
            
#         Returns:
#         --------
#         f(x) : `float`
#             Evaluating CENT objective function.
#         """   
#         return self.t*np.squeeze(0.5*x.T@self.P@x + self.q.T@x) - np.sum(np.log(self.h - self.G@x)) 
    
#     def df(self, x):
#         """
#         Parameters:
#         -----------
#         x : `numpy.ndarray`
#             (d,1)
            
#         Returns:
#         --------
#         df(x) : `numpy.ndarray`
#             (d+p,1) Evaluating first derivative of the CENT objective function.
#         """
#         tmp = 1 / (self.h - self.G@x)
#         return self.t*(self.P@x + self.q) + self.G.T@tmp
    
#     def d2f(self, x):
#         """
#         Parameters:
#         -----------
#         x : `numpy.ndarray`
#             (d,1)
            
#         Returns:
#         --------
#         d2f(x) : `numpy.ndarray`
#             (d+p,d+p) Evaluating second derivative of the CENT objective function.
#         """
#         tmp = 1 / (self.h - self.G@x)
#         return self.t*self.P + self.G.T@(np.diag(tmp)**2)@self.G   