#!/usr/bin/python

import numpy as np
from sys_function import sys_remove_modules

sys_remove_modules("extern_fn")
from extern_fn import *

class SelectParent(ExtenFn):
    """
    Clasa 'SelectParent', ofera doar metode pentru a selecta unul din parinti in calitate de parinte 1 sau 2
    Functia 'selectParent' nu are parametri.
    Pentru a folosi aceasta functie este necesar la inceputul fiecarei generatii de apelat functia 'startEpoch', cu parametrul 'fitness_values'.
    Metoda 'call', aplica functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self, extern_fn=None):
        super().__init__(extern_fn, "Metrics")
        self._extern_fn = self.__unpack(extern_fn)

    def __call__(self):
        return self._extern_fn()

    def __unpack(self, extern_fn):
        fn = self.selectParentAbstract
        self.startEpoch = self.startEpochAbstract
        if (extern_fn is not None):
            fn = extern_fn
            if (hasattr(extern_fn, "startEpoch")):
                print("startEpoch")
                self.startEpoch = extern_fn.startEpoch
            else:
                print("not startEpoch")
        return fn

    def selectParentAbstract(self, **kw):
        raise NameError("Functia 'SelectParent', lipseste metoda '{}', config: '{}'".format(self.__method, self.__config))

    def startEpochAbstract(self, **kw):
        raise NameError("Functia 'SelectParent', lipseste functia 'startEpoch' din extern '{}'".format(self._extern_fn))
