#!/usr/bin/python

import numpy as np

def normalization(x):
    x_min = x.min()
    x_max = x.max()
    denom = x_max - x_min
    if (denom == 0):
        # all values are identical: return a constant vector (e.g. all 1s)
        x_ret = np.ones_like(x, dtype=np.float32)
    else:
        x_ret = (x-x_min)/denom
    return x_ret

def min_nonzeronorm(x):
    mask_not_zero = (x!=0)
    valid_x = x[mask_not_zero]
    if (valid_x.shape[0] > 0):
        x_min = valid_x.min()
    else:
        x_min = 1
    return (2*x_min)/(x+x_min)

def min_nonzero(x):
    mask_not_zero = (x!=0)
    valid_x = x[mask_not_zero]
    if (valid_x.shape[0] > 0):
        x_min = valid_x.min()
    else:
        x_min = 1
    return x_min

def normalization_reference(x, x_min, x_max):
    denom = x_max - x_min
    if (denom == 0):
        # all values are identical: return a constant vector (e.g. all 1s)
        x_ret = np.ones_like(x, dtype=np.float32)
    else:
        x_ret = (x-x_min)/denom
    return x_ret

def min_nonzeronorm_reference(x, x_min):
    return (2*x_min)/(x+x_min)

def min_nonzero_values(metric_values, prev_metric_values, key):
    tmp      = metric_values[key]
    tmp_prev = prev_metric_values[key]
    tmp = np.concatenate((tmp, tmp_prev), axis=None)
    return min_nonzero(tmp)

def normal_values(metric_values, prev_metric_values, key):
    tmp      = metric_values[key]
    tmp_prev = prev_metric_values[key]
    tmp = np.concatenate((tmp, tmp_prev), axis=None)
    return tmp.min(), tmp.max()

def standart_values(metric_values, prev_metric_values, key):
    tmp      = metric_values[key]
    tmp_prev = prev_metric_values[key]
    tmp = np.concatenate((tmp, tmp_prev), axis=None)
    return tmp.mean(), tmp.std()

