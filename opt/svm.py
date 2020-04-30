import numpy as np
from scipy import stats
import cvxopt

from opt.algos.interior_point import barrier_method
from opt.utils.kernels import polynomial_kernel_matrix, gaussian_kernel_matrix

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
        
    def fit(self, X, y):
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
        if self.decision_function_shape == "ovo":
            for ClassVsClass, labels in y.items():
                data = X[ClassVsClass]
                # Get kernel matrix
                if self.kernel == "poly":
                    gamma = self.param
                    kernel_matrix = polynomial_kernel_matrix(data, data, 0, gamma)
                elif self.kernel == "gauss":
                    if self.param == "scale":
                        gamma = 1/(data.shape[1]*data.var())
                    elif self.param == "auto":
                        gamma = 1/data.shape[1]
                    else:
                        gamma = self.param
                    kernel_matrix = gaussian_kernel_matrix(data, data, gamma)
                # Fit classifiers
                self.classifiers[ClassVsClass] = SVM(self.C, self.kernel, gamma)
                self.opt_info[ClassVsClass] = self.classifiers[ClassVsClass].fit(data, labels, kernel_matrix, 
                                                                                 self.loss_fn, self.opt_algo)                   
                
        elif self.decision_function_shape == "ovr":
            # Get kernel matrix
            if self.kernel == "poly":
                kernel_matrix = polynomial_kernel_matrix(X, X, 0, self.param)
            elif self.kernel == "gauss":
                if self.param == "scale":
                    self.param = 1/(data.shape[1]*data.var())
                elif self.param == "auto":
                    self.param = 1/data.shape[1]  
                kernel_matrix = gaussian_kernel_matrix(X, X, self.param) 
            # Fit classifiers
            for ClassVsR, labels in y.items():
                self.classifiers[ClassVsR] = SVM(self.C, self.kernel, self.param)
                self.opt_info[ClassVsR] = self.classifiers[ClassVsR].fit(X, labels, kernel_matrix, 
                                                                         self.loss_fn, self.opt_algo)               
        
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
            mapping = dict()
            for i, (ClassVsR, svm) in enumerate(self.classifiers.items()):
                predictions[i,:] = svm.predict(X)
                mapping[i] = int(ClassVsR[0])
            # Return the label with the highest value.
            return np.array([mapping[i] for i in np.argmax(predictions, axis=0)])
        
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