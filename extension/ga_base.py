#!/usr/bin/python

import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import *

class GABase(RootGA):
    """
    Clasa 'GABase', 
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, name="GABase", **configs):
        super().__init__()
        self.__method = method
        self.__name   = name
        self._configs = configs

    def __call__(self, *args):
        raise NameError("Functia '{}', lipseste implementarea: '__call__'".format(self.__name))

    def __str__(self):
        info  = "{}: method '{}'\n".format(self.__name, self.__method)
        tmp   = "configs: '{}'\n".format(self._configs)
        info += "\t{}".format(tmp)
        return info

    def _unpackMethod(self, method, **kw):
        fn = None
        if (method is not None):
            fn = kw.get(method, None)
        # check exist function
        if (fn is None):
            fn = self.nullFn
        return fn

    def nullFn(self, *args, **kw):
        error_mesage = "Functia '{}', lipseste metoda '{}', config: '{}'\n".format(self.__name, self.__method, self._configs)
        raise NameError(error_mesage)
