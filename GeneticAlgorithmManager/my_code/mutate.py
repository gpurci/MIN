#!/usr/bin/python

import numpy as np
'''from sys_function import sys_remove_modules

sys_remove_modules("genom_op")'''
from genom_op import *

class Mutate(GenomOp):
    """
    Clasa 'Mutate', ofera doar metode pentru a face mutatia genetica a unui individ din populatie.
    Functia 'mutate' are 3 parametri, parinte1, parinte2, descendent.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, genome, method, **configs):
        super().__init__(genome, "Mutate", **configs)
        self.setFunctional(self.mutateChromosome, self.mutateGenome, self.mutateMixt)
        self.setAbstract(self.mutateAbstract, self.mutateChromosomeAbstract)
        self._unpackConfigs(method)

    def __call__(self, parent1, parent2, offspring, **kw):
        return self._man_fn(parent1, parent2, offspring, **kw)

    def mutateAbstract(self, *args, **kw):
        error_mesage = "Functia 'Mutate', lipseste metoda '{}', config: '{}'\n".format(self._method, self._configs)
        raise NameError(error_mesage)

    def mutateChromosomeAbstract(self, *args, **kw):
        error_mesage = "Functia 'Mutate', lipseste metoda '{}', chromozomul '{}', config: '{}'\n".format(self._method, self.__chromosome_name, self._configs)
        raise NameError(error_mesage)

    def __mutateChromosome(self, parent1, parent2, offspring, chrom_name, **kw):
        # calcularea ratei de probabilitate a mutatiei
        rate = np.random.uniform(low=0, high=1, size=None)
        if (rate <= self.MUTATION_RATE): # aplicarea operatiei de mutatie
            offspring = self._chromosome_fn[chrom_name](
                                                parent1[chrom_name], 
                                                parent2[chrom_name], 
                                                offspring[chrom_name], 
                                                **kw
                                            )
        else:
            offspring = offspring[chrom_name]

        return offspring

    def mutateChromosome(self, parent1, parent2, offspring, **kw):
        tmp_genome = []
        for self.__chromosome_name in self._genome.keys():
            chromosome_val = self.__mutateChromosome(parent1, parent2, offspring, self.__chromosome_name, **kw)
            tmp_genome.append(chromosome_val)
        return self._genome.concat(tmp_genome)

    def mutateGenome(self, parent1, parent2, offspring, **kw):
        # adaugare mutate rate
        rate = np.random.uniform(low=0, high=1, size=None)
        if (rate <= self.CROSSOVER_RATE): # operatia de incrucisare
            offspring = self._chromosome_fn["genome"](parent1, parent2, offspring, **kw)
        else: # selectie chromosom intreg
            pass
            #offspring = offspring
        return offspring

    def mutateMixt(self, parent1, parent2, offspring, **kw):
        error_mesage = "Lipseste metoda 'mutateMixt' functia de 'Mutate': config '{}'\n".format(self._configs)
        raise NameError(error_mesage)
