#!/usr/bin/python

import numpy as np
from extension.mutate.mutate_base import *

class MutateBinarySwap(MutateBase):
    """
    Clasa 'MutateBinarySwap', ofera doar metode pentru a face mutatia genetica a unui individ din populatie.
    Functia 'mutate' are 3 parametri, parinte1, parinte2, descendent.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, **configs):
        super().__init__(method, name="MutateBinarySwap", **configs)
        self.__fn = self._unpackMethod(method, 
                                        binary=self.mutateBinary, 
                                        swap=self.mutateSwap,
                                        mixt=self.mutateMixt)

    def __call__(self, parent1, parent2, offspring):
        return self.__fn(parent1, parent2, offspring, **self._configs)

    def help(self):
        info = """MutateBinarySwap:
    metoda: 'binary'; config: None;
    metoda: 'swap';   config: None;
    metoda: 'mixt';        config: -> p_select=[1/2, 1/2];\n"""
        print(info)

    def mutateBinary(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        # 
        locus = np.random.randint(low=0, high=self.GENOME_LENGTH, size=None)
        offspring[locus] = 1 - offspring[locus]
        return offspring

    def mutateSwap(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        # 
        locus1 = np.random.randint(low=0,                     high=self.GENOME_LENGTH//2, size=None)
        locus2 = np.random.randint(low=self.GENOME_LENGTH//2, high=self.GENOME_LENGTH,    size=None)
        offspring[locus1], offspring[locus2] = offspring[locus2], offspring[locus1]
        return offspring

    def mutateMixt(self, parent1, parent2, offspring, p_select=None):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        cond = np.random.choice([0, 1], size=None, p=p_select)
        if   (cond == 0):
            offspring = self.mutateBinary(parent1, parent2, offspring)
        elif (cond == 1):
            offspring = self.mutateSwap(parent1, parent2, offspring)
        return offspring
