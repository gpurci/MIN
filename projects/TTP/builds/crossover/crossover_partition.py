#!/usr/bin/python

import numpy as np

from sys_function import sys_remove_modules
sys_remove_modules("extension.crossover.ox_utils")

from extension.crossover.crossover_base import *
from extension.crossover.ox_utils import * 

class CrossoverPartition(CrossoverBase):  # TO DO:
    """
    Clasa 'CrossoverPartition', 
    Functia 'crossover' are 2 parametri, parinte1, parinte2.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, **configs):
        super().__init__(method, name="CrossoverPartition", **configs)
        self.__fn = self._unpackMethod(method, 
                                        scramble=self.crossoverScramble, 
                                        shift=self.crossoverShift,
                                        inversion=self.crossoverInversion,
                                        mixt=self.crossoverMixt)

    def __call__(self, parent1, parent2):
        return self.__fn(parent1, parent2, **self._configs)

    def help(self):
        info = """CrossoverPartition:
    metoda: 'scramble';  config None;
    metoda: 'shift';     config None;
    metoda: 'inversion'; config None;
    metoda: 'mixt';      config -> "p_select":[1/3, 1/3, 1/3], ;\n"""
        print(info)
