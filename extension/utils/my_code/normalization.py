#!/usr/bin/python

import numpy as np

def normalization(x):
    x_min = x.min()
    x_max = x.max()
    x_ret = (x_max-x)/(x_max-x_min+1e-7)
    return x_ret

def min_norm(x):
    mask_not_zero = (x!=0)
    valid_x = x[mask_not_zero]
    if (valid_x.shape[0] > 0):
        x_min = valid_x.min()
    else:
        x_min = 0.1
        x[:] = 0.1
    return (2*x_min)/(x+x_min)
