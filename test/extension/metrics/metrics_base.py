#!/usr/bin/python

import numpy as np
from extension.ga_base import *

class MetricsBase(GABase):
    """
    Clasa 'MetricsBase', 
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, name="MetricsBase", **configs):
        super().__init__(method, name=name, **configs)

    def _unpackMethod(self, method, **kw):
        fn, getScore = (None, None)
        if (method is not None):
            fn, getScore = kw.get(method, (None, None))
        # check exist function
        if (fn is None):
            fn = self.nullFn
            getScore = self.nullFn
        return fn, getScore

    def getArgBest(self, fitness_values):
        """Cautarea rutei optime din populatie"""
        index = np.argmax(fitness_values, axis=None, keepdims=False)
        return index
