#!/usr/bin/python

import numpy as np
from sys_function import sys_remove_modules

sys_remove_modules("genom_op")

from genom_op import *

class Crossover(GenomOp):
    """
    Clasa 'Crossover', ofera doar metode pentru a face incrucisarea genetica a doi parinti
    Functia 'crossover' are 2 parametri, parinte1, parinte2.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, genome, method, **configs):
        super().__init__(genome, "Crossover", **configs)
        self.setFunctional(self.crossoverChromosome, self.crossoverGenome, self.crossoverMixt)
        self.setAbstract(self.crossoverAbstract, self.crossoverChromosomeAbstract)
        self._unpackConfigs(method)

    def __call__(self, parent1, parent2, **kw):
        return self._man_fn(parent1, parent2, **kw)

    def crossoverAbstract(self, *args, **kw):
        error_mesage = "Functia 'Crossover', lipseste metoda '{}', config: '{}'\n".format(self._method, self._configs)
        raise NameError(error_mesage)

    def crossoverChromosomeAbstract(self, *args, **kw):
        error_mesage = "Functia 'Crossover', lipseste metoda '{}', chromozomul '{}', config: '{}'\n".format(self._method, self.__chromosome_name, self._configs)
        raise NameError(error_mesage)

    def __crossoverChromosome(self, parent1, parent2, chrom_name, **kw):
        # adaugare crossover rate
        rate = np.random.uniform(low=0, high=1, size=None)
        if (rate <= self.CROSSOVER_RATE): # operatia de incrucisare
            offspring = self._chromosome_fn[chrom_name](
                                                parent1[chrom_name], 
                                                parent2[chrom_name], 
                                                **kw
                                            )
        else: # selectie chromosom intreg
            if (self.__select_parent_chromosome == 0): # mosteneste chromosome parinte 1
                offspring = parent1[chrom_name].copy()
                self.__select_parent_chromosome = 1
            else:                                      # mosteneste chromosome parinte 2
                offspring = parent2[chrom_name].copy()
                self.__select_parent_chromosome = 0
        return offspring

    def crossoverChromosome(self, parent1, parent2, **kw):
        tmp_genome = []
        self.__select_parent_chromosome = 0
        for self.__chromosome_name in self._genome.keys():
            chromosome_val = self.__crossoverChromosome(parent1, parent2, self.__chromosome_name, **kw)
            tmp_genome.append(chromosome_val)
        return self._genome.concat(tmp_genome)

    def crossoverGenome(self, parent1, parent2, **kw):
        # adaugare crossover rate
        rate = np.random.uniform(low=0, high=1, size=None)
        if (rate <= self.CROSSOVER_RATE): # operatia de incrucisare
            offspring = self._chromosome_fn["genome"](parent1, parent2, **kw)
        else: # selectie chromosom intreg
            offspring = parent1.copy()
        return offspring

    def crossoverMixt(self, parent1, parent2, **kw):
        error_mesage = "Lipseste metoda 'crossoverMixt' functia de 'Crossover': config '{}'\n".format(self._configs)
        raise NameError(error_mesage)
