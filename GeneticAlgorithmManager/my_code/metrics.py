#!/usr/bin/python

import numpy as np
'''from sys_function import sys_remove_modules

sys_remove_modules("extern_fn")'''
from extern_fn import *

class Metrics(ExtenFn):
    """
    Clasa 'Metrics', 
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, extern_fn=None):
        super().__init__(extern_fn, "Metrics")
        self._extern_fn = self.__unpack(extern_fn)

    def __call__(self, genomics):
        return self._extern_fn(genomics)

    def __unpack(self, extern_fn):
        fn = self.metricsAbstract
        self.getScore = self.getScoreAbstract
        if (extern_fn is not None):
            fn = extern_fn
            if (hasattr(extern_fn, "getScore")):
                self.getScore = extern_fn.getScore
        return fn

    def metricsAbstract(self, *args):
        raise NameError("Functia 'Metrics', lipseste functia externa '{}'".format(self._extern_fn))

    def getScoreAbstract(self, *args):
        raise NameError("Functia 'Metrics', lipseste functia 'getScore' din extern '{}'".format(self._extern_fn))

