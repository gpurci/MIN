#!/usr/bin/python

import numpy as np
from my_code.root_GA import *

class IndividRepair(RootGA):
    """
    Clasa 'IndividRepair', ofera doar metode pentru a initializa populatia.
    Functia 'individRepair' are 1 parametru, individ - individul care va fi reparat.
    Metoda '__config_fn', selecteaza functia de initializare.
    Metoda '__call__', aplica functia de initializare ce a fost selectata in '__config_fn'
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self, config):
        self.setConfig(config)

    def __call__(self, size):
        return self.fn(size)

    def __config_fn(self):
        self.fn = self.individRepairAbstract
        if (self.__config is not None):
            if   (self.__config == "mixt"):
                self.fn = self.individRepairMixt
            elif (self.__config == "null"):
                self.fn = self.individRepairNull
        else:
            pass

    def setConfig(self, config):
        self.__config = config
        self.__config_fn()

    def individRepairAbstract(self, size):
        raise NameError("Lipseste configuratia pentru functia de 'IndividRepair': config '{}'".format(self.__config))

    def individRepairNull(self, individ):
        return individ

    def individRepairMixt(self, individ):# TO DO: aplica shift sau permutare pe secvente mai mici
        """Initializare individ, cu drumuri aleatorii si oras de start
        start_gene - orasul de start
        """
        raise NameError("Functia de 'individRepairMixt', incompleta") # TO DO
        cond = np.random.randint(low=0, high=2, size=None)
        size_shift = np.random.randint(low=1, high=TSP.GENOME_LENGTH-6, size=None)
        if (cond == 0):
            individ[1:-1] = np.roll(individ[1:-1], size_shift)
        else:
            args = np.random.choice(5, size=5, p=None)+size_shift
            individ[size_shift:size_shift+5] = individ[args]
        return individ
