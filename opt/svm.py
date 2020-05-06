import numpy as np
from scipy import stats
import cvxopt
import time

from opt.algos.interior_point import barrier_method
from opt.algos.smo import smo
from opt.utils.kernels import polynomial_kernel_matrix, gaussian_kernel_matrix
from opt.utils.feasible import feasible_starting_point

class SVC:
    """
    Multiclass soft-margin kernelised SVM.  
    """
    def __init__(self, C=1.0, kernel="gauss", param=0.5, decision_function_shape="ovo", 
                 loss_fn='L1', opt_algo="smo"):
        """
        Arguments:
        ----------
        C : `float`
            Regularization parameter. The strength of the regularization is 
            inversely proportional to C. Must be strictly positive. 
            The penalty is a squared l2 penalty.
        kernel : `str`
            The Kernel to use. {'poly', 'gauss'}.
        param : `float`
            Parameter of the kernel chosen, 
            i.e. the degree if polynomial or the gamma is gaussian. {'scale', 'auto'}
        decision_function_shape : `NoneType` or `str`
            The method for multiclass classification. {'ovo','ovr'}.
        loss_fn : `str`
            The loss function for the optimisation problem. {'L1','L2'}.
        opt_algo : `str`
            The optimisation method to use. {'barrier', 'smo'}.
        """         
        self.C = float(C)
        self.kernel = kernel
        self.param  = param
        self.decision_function_shape = decision_function_shape
        self.loss_fn  = loss_fn
        self.opt_algo = opt_algo
        self.classifiers = dict()
        self._params = dict()
        
    def fit(self, X, y, **kwargs):
        """
        Function to train the SVM.
        
        Parameters:
        -----------
        X : `dict` of `numpy.ndarray` or `numpy.ndarray`
             if self.decision_function_shape="ovo":
                 Dictionary of all the kchoose2 datasets for the classifiers.
                (nData, nDim) matrix of data.
             if self.decision_function_shape="ovr"
                 (nData, nDim) matrix of data.
        y : `dict` of `numpy.ndarray`
             if self.decision_function_shape="ovo":
                 Dictionary of labels for the kchoose2 classifiers.
                (nData,) matrix of corresponding labels. Each element is in the set {-1,+1}.
             if self.decision_function_shape="ovr"
                 Dictionary of labels for the k classifiers.
                (nData,) matrix of corresponding labels. Each element is in the set {-1,+1}.
        """
        self.opt_info = dict()
        start_time = time.time()
        if self.decision_function_shape == "ovo":
            for ClassVsClass, labels in y.items():
                data = X[ClassVsClass]
                # Get kernel matrix
                if self.kernel == "poly":
                    self._params[ClassVsClass] = self.param
                    kernel_matrix = polynomial_kernel_matrix(data, data, 0, self._params[ClassVsClass])
                elif self.kernel == "gauss":
                    if self.param == "scale":
                        self._params[ClassVsClass] = 1/(data.shape[1]*data.var())
                    elif self.param == "auto":
                        self._params[ClassVsClass] = 1/data.shape[1]
                    else:
                        self._params[ClassVsClass] = self.param 
                    kernel_matrix = gaussian_kernel_matrix(data, data, self._params[ClassVsClass])
                # Initialise binary classifier
                self.classifiers[ClassVsClass] = SVM(self.C, self.kernel, self._params[ClassVsClass])
                # Fit binary classifier
                self.opt_info[ClassVsClass] = self.classifiers[ClassVsClass].fit(data, labels, kernel_matrix, 
                                                                                 self.loss_fn, self.opt_algo,
                                                                                 **kwargs)                   
                
        elif self.decision_function_shape == "ovr":
            # Get kernel matrix
            if self.kernel == "poly":
                self._params[ClassVsR] = self.param                                    
                kernel_matrix = polynomial_kernel_matrix(X, X, 0, self._params[ClassVsR])
            elif self.kernel == "gauss":
                if self.param == "scale":
                    self._params[ClassVsR] = 1/(data.shape[1]*data.var())
                elif self.param == "auto":
                    self._params[ClassVsR] = 1/data.shape[1]
                else:
                    self._params[ClassVsR] = self.param
                kernel_matrix = gaussian_kernel_matrix(X, X, self._params[ClassVsR]) 
            for ClassVsR, labels in y.items():
                # Initialise binary classifier
                self.classifiers[ClassVsR] = SVM(self.C, self.kernel, self._params[ClassVsR])
                # Fit binary classifier
                self.opt_info[ClassVsR] = self.classifiers[ClassVsR].fit(X, labels, kernel_matrix, 
                                                                         self.loss_fn, self.opt_algo, 
                                                                         **kwargs) 
        self.time_taken = round(time.time() - start_time, 6)
        
    def predict(self, X):
        """
        Predict on test set.
        
        Parameters:
        -----------
        X : `numpy.ndarray`
            (nData, nDim) matrix of test data. Each row corresponds to a data point.  
            
        Returns:
        --------
        yhats : `numpy.ndarray`
            (nData,) List of predictions on the test data.
        """
        nClassifiers = len(self.classifiers)
        # Create matrix (nClassifiers, nTestData) 
        predictions = np.zeros((nClassifiers, X.shape[0]))
        
        if self.decision_function_shape == "ovo":
            for i, (ClassVsClass, svm) in enumerate(self.classifiers.items()):
                yhats = np.sign(svm.predict(X))
                # Convert predictions labels to digits.
                predictions[i,:] = self.convert_labels2digits(yhats, int(ClassVsClass[0]), int(ClassVsClass[-1]))
            # Return the mode label for each test data point.
            # If there is more than one such value, only the smallest is returned.
            return np.squeeze(stats.mode(predictions, axis=0)[0])
       
        elif self.decision_function_shape == "ovr":
            index2class = dict()
            for i, (ClassVsR, svm) in enumerate(self.classifiers.items()):
                predictions[i,:] = svm.predict(X)
                index2class[i] = int(ClassVsR[0])
            # Return the label with the highest value.
            return np.array([index2class[i] for i in np.argmax(predictions, axis=0)])
        
    @staticmethod
    def convert_labels2digits(yhats, pos_label, neg_label):
        """
        Functions that maps +1 to the positive class label 
        and -1 to the negative class label.
        
        Parameters:
        -----------
        yhats : `numpy.ndarray`
            The predictions from the classifier.
        pos_label : `int`
            The positive class label.
        neg_label : `int`
            The negative class label.
            
        Returns:
        --------
        digits : `numpy.ndarray`
            The predicitions mapped to the class labels +1 -> pos_labels, -1 -> neg_labels.
        """
        return np.where(yhats==1, pos_label, neg_label)        
    
class SVM:
    """
    Soft-margin kernalised SVM base class
    """
    def __init__(self, C=1.0, kernel='gauss', param=0.5):
        """
        Arguments:
        ----------
        C : `float`
            Regularization parameter. The strength of the regularization is 
            inversely proportional to C. Must be strictly positive. 
            The penalty is a squared l2 penalty.  
        kernel : `str`
            The Kernel to use. {'poly', 'gauss'}.      
        param : `float`
            Parameter of the kernel chosen, 
            i.e. the degree if polynomial or the gamma is gaussian.             
        """        
        self.C = float(C)
        self.kernel = kernel
        self.param  = param
        
    def fit(self, X, y, kernel_matrix, loss_fn, opt_algo, **kwargs):
        """
        Function to fit SVM.
        
        min 0.5x^TPx + q^Tx
        s.t. Gx <= h
             Ax = b        
        
        Parameters:
        -----------
        X : `numpy.ndarray`
            (nData, nDim) matrix of data. Each row corresponds to a data point.
        y : `numpy.ndarray`
            (nData,) matrix of corresponding labels. Each element is in the set {-1,+1}.
        kernel_matrix : `numpy.ndarray`
            (nData, nData) Kernel matrix of training data.  
        loss_fn : `str`
            The loss function for the optimisation problem. {'L1','L2'}.            
        opt_algo : `str`
            The optimisation method to use. {'barrier', 'smo'}.
            
        Returns:
        --------
        opt_info : `dict`
            Information about optimisation.            
        """ 
        # Convert data to floats
        X = X.astype(float)
        y = y[:,None].astype(float)
        n, d = X.shape
        
        # Parameters of optimisation problem
        if loss_fn == 'L1':
            P = y@y.T*kernel_matrix
            q = -np.ones((n,1))
            G = np.vstack((-np.eye(n), 
                            np.eye(n)))
            h = np.vstack((np.zeros((n,1)), 
                           self.C*np.ones((n,1))))
            A = y.T
            b = np.zeros(1)    
        elif loss_fn == 'L2':
            P = y@y.T*(kernel_matrix + 1/self.C * np.eye(n))
            q = -np.ones((n,1))
            G = -np.eye(n)
            h = np.zeros((n,1))
            A = y.T
            b = np.zeros(1)
        
        # Optimisation        
        if opt_algo == "barrier":      
            # Get feasible starting point
            x0  = feasible_starting_point(y, self.C)  
            # Solve QP problem
            sol = barrier_method(P, q, G, h, A, b, x0, **kwargs) 
            # Get lambdas (the Lagrange multipliers)
            lambdas = sol['x']
            
        elif opt_algo == "smo":     
            # Solve QP problem
            sol = smo(y, kernel_matrix, self.C, loss_fn, **kwargs) 
            # Get lambdas (the Lagrange multipliers)
            lambdas = sol['x']
        
        elif opt_algo == "cvxopt":
            P = cvxopt.matrix(P)
            q = cvxopt.matrix(q)
            G = cvxopt.matrix(G)
            h = cvxopt.matrix(h)
            A = cvxopt.matrix(A)
            b = cvxopt.matrix(b)         
            # Setting solver parameters
            cvxopt.solvers.options['show_progress'] = False   
            # Solve QP problem
            sol = cvxopt.solvers.qp(P, q, G, h, A, b)
            # Get lambdas (the Lagrange multipliers)
            lambdas = np.array(sol['x'])
              
        # Support vectors have non-zero lambdas
        if loss_fn == 'L1':
            SV  = np.logical_and(lambdas>1e-5, lambdas<=self.C).flatten() # Margin and non-margin support vectors
            MSV = np.logical_and(lambdas>1e-5, lambdas<self.C).flatten()  # Margin support vectors
        elif loss_fn == 'L2':
            SV  = (lambdas>1e-5).flatten()
        self.lambdas = lambdas[SV]
        self.sv   = X[SV]
        self.sv_y = y[SV]  
        sol["nSVs"] = sum(SV) 
        # Compute bias term
        if loss_fn == "L1":
            self.bias = np.mean(y[SV] - ((lambdas[SV]*y[SV]).T@kernel_matrix[SV][:,SV]).T) 
        elif loss_fn == "L2":
            self.bias = np.mean(y[SV] - ((lambdas[SV]*y[SV]).T@(kernel_matrix[SV][:,SV]+(1/self.C)*np.eye(sol["nSVs"]))).T)
        return sol

    def predict(self, X):
        """
        Predict on test set.
        
        Parameters:
        -----------
        X : `numpy.ndarray`
            (nData, nDim) matrix of test data. Each row corresponds to a data point.              
            
        Returns:
        --------
        yhats : `numpy.ndarray`
            (nData,) List of predictions on the test data, not corresponding to labels {-1,+1}
            Need to take np.sign() after.
        """               
        # Get kernel matrix between support vectors and test data
        if self.kernel == "poly":
            kernel_matrix = polynomial_kernel_matrix(self.sv, X, 0, self.param)
        elif self.kernel == "gauss":
            kernel_matrix = gaussian_kernel_matrix(self.sv, X, self.param) 
        return np.squeeze((self.lambdas*self.sv_y).T@kernel_matrix + self.bias)    