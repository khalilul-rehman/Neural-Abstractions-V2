import numpy as np
import pandas as pd


def normalized_root_mean_square_error(y_true, y_pred):
    """
    Computes the Normalized Root Mean Square Error (NRMSE) between y_true and y_pred.
    If the range of y_true is zero, it normalizes by the number of samples * outputs.

    Parameters:
        y_true (np.ndarray): Ground truth values, shape (n_samples, n_outputs)
        y_pred (np.ndarray): Predicted values, shape (n_samples, n_outputs)

    Returns:
        float: NRMSE value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Compute range
    y_range = np.max(y_true) - np.min(y_true)
    
    if y_range != 0:
        # Normalize by range
        return rmse / y_range
    else:
        # Normalize by n_samples * n_outputs
        n_samples, n_outputs = y_true.shape
        return np.sqrt(np.sum((y_true - y_pred) ** 2) / (n_samples * n_outputs))


# Function to load DataSet
def load_dataset(file_path, num_attributes=2, num_classes=2):
    data = pd.read_csv(file_path)
    X = data.iloc[:, 0 :  num_attributes].values
    y = data.iloc[:,  num_attributes:  num_attributes + num_classes].values
    # y = data.iloc[:, 9:10].values
    return X, y




# Function to get the indicies on different leaves
def get_leaf_samples(tree, X):
    """Return dictionary: leaf_id -> sample indices"""
    leaf_indices = tree.apply(X)
    leaf_samples = {}
    for i, leaf_id in enumerate(leaf_indices):
        if leaf_id not in leaf_samples:
            leaf_samples[leaf_id] = []
        leaf_samples[leaf_id].append(i)
    return  leaf_samples.items()