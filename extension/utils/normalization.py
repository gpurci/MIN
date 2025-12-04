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
        x_ret = (x_max-x)/denom
    return x_ret

def min_nonzeronorm(x):
    mask_not_zero = (x!=0)
    valid_x = x[mask_not_zero]
    if (valid_x.shape[0] > 0):
        x_min = valid_x.min()
    else:
        x_min = 1
    return (2*x_min)/(x+x_min)
