import numpy as np
from mnist import MNIST
from collections import defaultdict

# from opt.utils.preprocess import process_raw_data
# process_raw_data(input_filepath='data/mnist/', 
#                  output_filepath='data/filtered_mnist', 
#                  classes2keep=[0,1,6,9])

def process_raw_data(input_filepath, output_filepath, classes2keep=None):
    """
    Function to process the raw data
    
    Parameters:
    -----------
    input_filepath : `str`
        The input filepath to the dataset.
    output_filepath : `str`
        The output filepath to save the processed dataset.
    classes2keep : `Nonetype` or `list`
        The class labels to keep.
    """
    # Load MNIST
    mnist = MNIST(input_filepath)
    x_train, y_train = mnist.load_training()   #60000 samples
    x_test, y_test   = mnist.load_testing()    #10000 samples
    
    x_train = np.asarray(x_train).astype(np.float32) 
    y_train = np.asarray(y_train).astype(np.int32)
    x_test  = np.asarray(x_test).astype(np.float32)
    y_test  = np.asarray(y_test).astype(np.int32)   
    
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255    
    
    # Filter the dataset
    if classes2keep is not None:
        x_train, y_train = remove_classes(x_train, y_train, classes2keep)
        x_test, y_test   = remove_classes(x_test, y_test, classes2keep)
        
    # Save data 
    np.savez_compressed(output_filepath, 
                        a=x_train, 
                        b=y_train,
                        c=x_test,
                        d=y_test)   

def remove_classes(data, labels, classes2keep):
    """
    Function to cut dataset to keep only a few classes.
    
    Paramaters:
    -----------
    data : `numpy.ndarray`
        (nData, nDim) The dataset.
    labels : `numpy.ndarray`
        (nData,) The corresponding labels.
    classes2keep : `list`
        The class labels to keep.
    
    Returns:
    --------
    data : `numpy.ndarray`
        (nData, features) The filtered dataset.
    labels : `numpy.ndarray`
        (nData,) The corresponding labels.        
    """
    new_data = defaultdict(list)
    for i, label in enumerate(labels):
        if label in classes2keep:
            new_data["label"].append(label)
            new_data["data"].append(data[i])
    return np.array(new_data["data"]), np.array(new_data["label"])