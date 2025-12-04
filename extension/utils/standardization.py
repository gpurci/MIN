#!/usr/bin/python

import numpy as np

def standardization(x):
    x_mean = x.mean()
    x_std  = x.std()
    return (x - x_mean)/x_std