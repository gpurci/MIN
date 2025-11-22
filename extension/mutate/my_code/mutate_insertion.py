#!/usr/bin/python

import numpy as np
from extension.mutate.my_code.mutate_base import *

class MutateInsertion(MutateBase):
    """
    Clasa 'Mutate', ofera doar metode pentru a face mutatia genetica a unui individ din populatie.
    Functia 'mutate' are 3 parametri, parinte1, parinte2, descendent.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, **configs):
        super().__init__(method, name="MutateInsertion", **configs)
        self.__fn = self._unpackMethod(method, 
                                        insertion=self.mutateInsertion, )

    def __call__(self, parent1, parent2, offspring):
        return self.__fn(parent1, parent2, offspring, **self._configs)

    def help(self):
        info = """MutateInsertion:
    metoda: 'insertion'; config: None;\n"""
        print(info)

    def mutateInsertion(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        locus1 = np.random.randint(low=0,      high=self.GENOME_LENGTH//2, size=None)
        locus2 = np.random.randint(low=locus1, high=self.GENOME_LENGTH,    size=None)
        # copy gene
        gene1  = offspring[locus1]
        # make change locuses
        locuses= np.arange(locus1, locus2)
        offspring[locuses] = offspring[locuses+1]
        offspring[locus2]  = gene1
        return offspring
