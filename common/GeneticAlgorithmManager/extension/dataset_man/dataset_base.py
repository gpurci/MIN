#!/usr/bin/python

import numpy as np

class DatasetBase():
    """
    """
    def __init__(self, dataset, name):
        self.dataset = dataset
        self.__name  = name
 
    def __str__(self):
        info = "{}, GENOME_LENGTH: ".format(self.__name, self.dataset["GENOME_LENGTH"])
        return info


    def summary(self, **kw): # TO DO
        print("Summary")
        for name in kw.keys():
            val = kw[name]
            print("{}: min {}, max {}, mean {}, std {}".format(name, val.min(), val.max(), np.mean(val), np.std(val)))
