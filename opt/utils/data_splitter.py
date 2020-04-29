import numpy as np
from scipy.special import comb

def label2binary(y, label):
    """
    Map label val to +1 and the other labels to -1.
    
    Paramters:
    ----------
    y : `numpy.ndarray`
        (nData,) The labels of two classes.
    val : `int`
        The label to map to +1.
        
    Returns:
    --------
    y : `numpy.ndarray`
        (nData,) Maps the val label to +1 and the other label to -1.
    """
    return (2*(y == label).astype(int))-1 

def split4ovo(X, y):
    """
    Split the data into kchoose2 datasets
    
    Paramters:
    ----------
    X : `numpy.ndarray`
        (nData, nDim) The training data.
    y : `numpy.ndarray`
        (nData,) Corresponding training labels.       

    Returns:
    --------
    X_ovo : `dict` of `numpy.ndarray`
        Dictionary of datasets for one vs one mutliclass classification.
    y_ovo : `dict` of `numpy.ndarray`
        Dictionary of corresponding labels for one vs one mutliclass classification.    
    """
    X_ovo = dict()
    y_ovo = dict()
    unique_classes = np.unique(y).astype(int)
    nClasses = len(unique_classes)
    for i in unique_classes:
        unique_classes = np.delete(unique_classes, np.where(unique_classes == i), axis=0)
        for j in unique_classes:
            # Get indices for which labels are i vs j
            indices = np.logical_or(y==i, y==j)
            # Get trainxs from indices
            X_ovo[str(i)+"vs"+str(j)] = X[indices]
            # Get corresponding labels, i->1 and j->-1
            y_ovo[str(i)+"vs"+str(j)] = label2binary(y[indices], i)
    return X_ovo, y_ovo

def split4ovr(y):
    """
    Change the labels such that we map the class label that is the
    one in one vs all to +1 and the all to -1.
    
    Paramters:
    ----------
    y : `numpy.ndarray`
        (nData,) Corresponding training labels.       

    Returns:
    --------
    y_ovr : `dict` of `numpy.ndarray`
        Dictionary of labels for one vs rest mutliclass classification.      
    """
    unique_classes = np.unique(y).astype(int)
    y_ovr = dict()
    for i in unique_classes:
        y_ovr[str(i)] = label2binary(y, i)
    return y_ovr