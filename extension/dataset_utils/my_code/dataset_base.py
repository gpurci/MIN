#!/usr/bin/python

import numpy as np

class DatasetBase():
    """
    """
    def __init__(self, dataset):
    	self.dataset = dataset

    def computeIndividDistance(self, individ):
        d = self.dataset["distance"]
        return d[individ[:-1], individ[1:]].sum() + d[individ[-1], individ[0]]

    def individCityDistance(self, individ):
        d = self.dataset["distance"]
        city_distances = d[individ[:-1], individ[1:]]
        to_first_city  = d[individ[-1], individ[0]]
        return np.concatenate((city_distances, [to_first_city]))

    def neighbors(self, size):
    	genom_length  = self.dataset["GENOME_LENGTH"]
    	distances     = self.dataset["distance"]
    	x_range       = np.arange(genom_length, dtype=np.int32)
    	ret_neighbors = np.argsort(distances[x_range], axis=-1)[:, 1:size+1]
    	return ret_neighbors


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
