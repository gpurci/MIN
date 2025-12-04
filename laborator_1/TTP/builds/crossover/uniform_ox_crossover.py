#!/usr/bin/python

import numpy as np

from sys_function import sys_remove_modules
sys_remove_modules("extension.crossover.my_code.ox_utils")

from extension.crossover.my_code.crossover_base import *
from extension.crossover.my_code.ox_utils import * 

class CrossoverOXUnif(CrossoverBase):
    """
    Clasa 'CrossoverOXUnif', 
    Functia 'crossover' are 2 parametri, parinte1, parinte2.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, **configs):
        super().__init__(method, name="CrossoverOXUnif", **configs)
        self.__fn = self._unpackMethod(method, 
                                        scramble=self.crossoverScramble, 
                                        shift=self.crossoverShift,
                                        inversion=self.crossoverInversion,
                                        mixt=self.crossoverMixt)

    def __call__(self, parent1, parent2):
        return self.__fn(parent1, parent2, **self._configs)

    def help(self):
        info = """CrossoverOXUnif:
    metoda: 'scramble';  config None;
    metoda: 'shift';     config None;
    metoda: 'inversion'; config None;
    metoda: 'mixt';      config -> "p_select":[1/3, 1/3, 1/3], ;\n"""
        print(info)

    def crossoverScramble(self, parent1, parent2):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        locus   = np.random.choice(self.GENOME_LENGTH, size=self.GENOME_LENGTH//2, replace=False)
        parent2 = sim_scramble(parent1, parent2)
        return ox_crossover_order_parent2(parent1, parent2, locus)

    def crossoverShift(self, parent1, parent2):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        locus   = np.random.choice(self.GENOME_LENGTH, size=self.GENOME_LENGTH//2, replace=False)
        parent2 = sim_shift(parent1, parent2)
        return ox_crossover_order_parent2(parent1, parent2, locus)

    def crossoverInversion(self, parent1, parent2):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        locus   = np.random.choice(self.GENOME_LENGTH, size=self.GENOME_LENGTH//2, replace=False)
        parent2 = sim_inversion(parent1, parent2)
        return ox_crossover_order_parent2(parent1, parent2, locus)

    def crossoverMixt(self, parent1, parent2, p_select=None):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        cond = np.random.choice([0, 1, 2], size=None, p=p_select)
        if   (cond == 0):
            offspring = self.crossoverScramble( parent1, parent2)
        elif (cond == 1):
            offspring = self.crossoverShift(    parent1, parent2)
        elif (cond == 2):
            offspring = self.crossoverInversion(parent1, parent2)
        return offspring
