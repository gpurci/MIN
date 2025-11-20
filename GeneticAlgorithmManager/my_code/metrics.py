#!/usr/bin/python

import numpy as np
from root_GA import *

class Metrics():
    """
    Clasa 'Metrics', ofera doar metode pentru a calcula metrici pentru clase de probleme de optimizare.
    Functia 'metrics' are 1 parametru, populatia.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, extern_fn=None):
        super().__init__(extern_fn, "Metrics")

    def __call__(self, genomics):
        return self.__extern_fn(genomics)

    def __unpack(self, extern_fn):
        fn = self.metricsAbstract
        self.getScore = self.getScoreAbstract
        if (extern_fn is not None):
            fn = extern_fn
            if (hasattr(extern_fn, "getScore")):
                self.getScore = self.getScore
        return fn

    def metricsAbstract(self, *args):
        raise NameError("Functia 'Metrics', lipseste functia externa '{}'".format(self.__extern_fn))

    def getScoreAbstract(self, *args):
        raise NameError("Functia 'Metrics', lipseste functia 'getScore' din extern '{}'".format(self.__extern_fn))
