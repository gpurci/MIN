#!/usr/bin/python

import numpy as np
from extension.crossover.my_code.crossover_base import *
from extension.crossover.my_code.ox_utils import * 

class CrossoverOXSPUnif(CrossoverBase):
    """
    Clasa 'CrossoverOXSPUnif', 
    Functia 'crossover' are 2 parametri, parinte1, parinte2.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, **configs):
        super().__init__(method, name="CrossoverOXSPUnif", **configs)
        self.__fn = self._unpackMethod(method, 
                                        scramble=self.crossoverScramble, 
                                        shift=self.crossoverShift,
                                        inversion=self.crossoverInversion,
                                        mixt=self.crossoverMixt)

    def __call__(self, parent1, parent2):
        return self.__fn(parent1, parent2, **self._configs)

    def help(self):
        info = """CrossoverOXSPUnif:
    metoda: 'scramble';  config -> "subset_size":20;
    metoda: 'shift';     config -> "subset_size":20;
    metoda: 'inversion'; config -> "subset_size":20;
    metoda: 'mixt';      config -> "p_select":[1/3, 1/3, 1/3], "subset_size":20;\n"""
        print(info)

    def crossoverScramble(self, parent1, parent2, subset_size=20):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        # selectarea diapazonului de mostenire
        start = np.random.randint(low=0, high=self.GENOME_LENGTH-subset_size, size=None)
        # creare locus
        locus = np.arange(start, start+subset_size)
        parent2 = sim_scramble_field(parent1, parent2, start, subset_size, self.GENOME_LENGTH)
        return ox_crossover_order_parent2(parent1, parent2, locus)

    def crossoverShift(self, parent1, parent2, subset_size=20):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        # selectarea diapazonului de mostenire
        start = np.random.randint(low=0, high=self.GENOME_LENGTH-subset_size, size=None)
        # creare locus
        locus = np.arange(start, start+subset_size)
        parent2 = sim_shift_field(parent1, parent2, start, subset_size, self.GENOME_LENGTH)
        return ox_crossover_order_parent2(parent1, parent2, locus)

    def crossoverInversion(self, parent1, parent2, subset_size=20):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        # selectarea diapazonului de mostenire
        start = np.random.randint(low=0, high=self.GENOME_LENGTH-subset_size, size=None)
        # creare locus
        locus = np.arange(start, start+subset_size)
        parent2 = sim_inversion_field(parent1, parent2, start, subset_size, self.GENOME_LENGTH)
        return ox_crossover_order_parent2(parent1, parent2, locus)

    def crossoverMixt(self, parent1, parent2, p_select=None, subset_size=20):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        cond = np.random.choice([0, 1, 2], size=None, p=p_select)
        if   (cond == 0):
            offspring = self.crossoverScramble(parent1, parent2)
        elif (cond == 1):
            offspring = self.crossoverShift(parent1, parent2)
        elif (cond == 2):
            offspring = self.crossoverInversion(parent1, parent2)
        return offspring
