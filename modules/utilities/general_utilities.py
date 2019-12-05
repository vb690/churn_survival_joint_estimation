import numpy as np

import tensorflow.keras.backend as K


def load_arrays(arrays, dir_name):
    '''
    Function for loading saved arrays, this assume the existence of a local
    directory named data

    Arguments:
        - arrays: a list of strings, specifying the name of the arrays
                  to be loaded
        - dir_name: a string, specifying the directory where the
                    arrays are locatdd

    Returs:
        - loaded_arrays: a dictionry where keys are the identifiers of the
                         arrays and value are the loaded arrays
    '''
    load_dir = 'data\\{}'.format(dir_name)
    loaded_arrays = {}
    for array in arrays:

        path = '{}\\{}.npy'.format(load_dir, array)
        loaded_array = np.load(path, allow_pickle=True)
        loaded_arrays[array] = loaded_array

    return loaded_arrays


def smape_np(y_true, y_pred):
    '''
    Function for computing the Simmetric Mean Absolute Error (SMAPE)
    given numpy array:

    Args:
        - y_true: a numpy array, is the collection of ground truth values
        - y_pred: a numpy array, is the collection of predicted values

    Returns:
        - division: is a float, the SMAPE between y_true and y_pred
    '''
    nominator = np.sum(np.abs(y_pred - y_true))
    denominator = np.sum(y_true + y_pred)
    division = nominator / denominator
    return division


def smape_k(y_true, y_pred):
    '''
    Function for computing the Simmetric Mean Absolute Error (SMAPE)
    given keras tensors:

    Args:
        - y_true: a keras tensor, is the collection of ground truth values
        - y_pred: a keras tensor, is the collection of predicted values

    Returns:
        - division: is a float, the SMAPE between y_true and y_pred
    '''
    nominator = K.sum(K.abs(y_pred - y_true))
    denominator = K.sum(y_true + y_pred)
    division = nominator / denominator
    return division
