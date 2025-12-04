#!/usr/bin/python

import numpy as np
from extension.crossover.my_code.crossover_base import *

class CrossoverOXInversion(CrossoverBase):
    """
    Clasa 'CrossoverOXInversion', 
    Functia 'crossover' are 2 parametri, parinte1, parinte2.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, **configs):
        super().__init__(method, name="CrossoverOXInversion", **configs)
        self.__fn = self._unpackMethod(method, 
                                        in_space=self.crossoverIn, 
                                        out_space=self.crossoverOut,
                                        uniform=self.crossoverUniform,
                                        mixt=self.crossoverMixt)

    def __call__(self, parent1, parent2):
        return self.__fn(parent1, parent2, **self._configs)

    def help(self):
        info = """CrossoverOXInversion:
    metoda: 'in_space';  config None;
    metoda: 'out_space'; config None;
    metoda: 'uniform';   config None;
    metoda: 'mixt';      config -> "p_select":[1/3, 1/3, 1/3], ;\n"""
        print(info)

    def __crossoverOrderParent2(self, parent1, parent2, locus):
        # mosteneste parinte1
        offspring = parent1.copy()
        # obtinerea genelor de pe locus parent1
        genes_p1    = parent1[locus]
        # gasirea locusurilor genelor din parent 1 in parent2 
        _, pos_p2   = np.nonzero(parent2 == genes_p1.reshape(-1, 1))
        # aranjarea genelor dupa ordinea din parent 2
        sort_pos_p2 = np.argsort(pos_p2)
        # salvarea genelor dupa ordinea din parent 2
        offspring[locus] = genes_p1[sort_pos_p2]
        return offspring

    def __inversion(self, parent1, parent2):
        # check diff
        mask_sim_locus = parent1==parent2
        if (mask_sim_locus.sum() == mask_sim_locus.shape[0]):
            parent2 = np.flip(parent2)
        return parent2

    def crossoverIn(self, parent1, parent2):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        parent2 = self.__inversion(parent1, parent2)
        # selectarea diapazonului de mostenire
        start = np.random.randint(low=0,       high=self.GENOME_LENGTH//2, size=None)
        end   = np.random.randint(low=start+1, high=self.GENOME_LENGTH-1,  size=None)
        # creare locus
        locus = np.arange(start, end)
        return self.__crossoverOrderParent2(parent1, parent2, locus)

    def crossoverOut(self, parent1, parent2):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        parent2 = self.__inversion(parent1, parent2)
        # selectarea diapazonului de mostenire
        start = np.random.randint(low=0,       high=self.GENOME_LENGTH//2, size=None)
        end   = np.random.randint(low=start+1, high=self.GENOME_LENGTH-1,  size=None)
        # creare locus
        mask  = np.ones(self.GENOME_LENGTH, dtype=bool)
        mask[start:end] = False
        locus = np.argwhere(mask).reshape(-1)
        return self.__crossoverOrderParent2(parent1, parent2, locus)

    def crossoverUniform(self, parent1, parent2):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        parent2 = self.__inversion(parent1, parent2)
        # mosteneste parinte1
        offspring = parent1.copy()
        # selectarea diapazonului de mostenire
        locus = np.random.randint(low=0, high=self.GENOME_LENGTH, size=self.GENOME_LENGTH//2)
        return self.__crossoverOrderParent2(parent1, parent2, locus)

    def crossoverMixt(self, parent1, parent2, p_select=None):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        cond = np.random.choice([0, 1, 2], size=None, p=p_select)
        if   (cond == 0):
            offspring = self.crossoverIn(parent1, parent2)
        elif (cond == 1):
            offspring = self.crossoverOut(parent1, parent2)
        elif (cond == 2):
            offspring = self.crossoverUniform(parent1, parent2)
        return offspring
