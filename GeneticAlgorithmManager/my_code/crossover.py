#!/usr/bin/python

import numpy as np
from root_GA import *

class Crossover(RootGA):
    """
    Clasa 'Crossover', ofera doar metode pentru a face incrucisarea genetica a doi parinti
    Functia 'crossover' are 2 parametri, parinte1, parinte2.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, genome, **chromosom_configs):
        super().__init__()
        self.__genome = genome
        self.__chromosom_configs = chromosom_configs
        self.__unpackConfigs()

    def __call__(self, parent1, parent2):
        tmp_genome = []
        self.__select_parent_chromosome = 0
        for idx, chromozome_name in enumerate(self.__genome.keys(), 0):
            chromosome_val = self.__crossover_chromosome(parent1, parent2, chromozome_name)
            tmp_genome.append(chromosome_val)
        return self.__genome.concat(tmp_genome)

    def __crossover_chromosome(self, parent1, parent2, chromozome_name):
        # adaugare crossover rate
        rate = np.random.uniform(low=0, high=1, size=None)
        if (rate <= self.CROSSOVER_RATE): # operatia de incrucisare
            low, high = self.__genome.getGeneRange(chromozome_name)
            offspring = self.__fn[chromozome_name](parent1[chromozome_name], parent2[chromozome_name], 
                                        low, high, 
                                        **self.__chromosom_configs[chromozome_name])
        else: # selectie chromosom intreg
            if (self.__select_parent_chromosome == 0): # mosteneste chromosome parinte 1
                offspring = parent1[chromozome_name].copy()
                self.__select_parent_chromosome = 1
            else: # mosteneste chromosome parinte 2
                offspring = parent2[chromozome_name].copy()
                self.__select_parent_chromosome = 0
        return offspring

    def __str__(self):
        info = "Crossover:\n"
        for chrom_name in self.__genome.keys():
            tmp   = "Chromozome name: '{}', method '{}', configs: '{}'\n".format(chrom_name, self.__methods[chrom_name], self.__chromosom_configs[chrom_name])
            info += "\t{}".format(tmp)
        return info

    def __unpackMethod(self, method, extern_fn):
        fn = self.crossoverAbstract
        if (method is not None):
            if   (method == "diff"):
                fn = self.crossoverDiff
            elif (method == "split"):
                fn = self.crossoverSplit
            elif (method == "perm_sim"):
                fn = self.crossoverPermSim
            elif (method == "flip_sim"):
                fn = self.crossoverFlipSim
            elif (method == "mixt"):
                fn = self.crossoverMixt
            elif ((method == "extern") and (extern_fn is not None)):
                fn = extern_fn

        return fn

    def help(self):
        info = """Crossover:
    metoda: 'diff';     config None;
    metoda: 'split';    config None;
    metoda: 'perm_sim'; config None;
    metoda: 'flip_sim'; config None;
    metoda: 'mixt';     config -> "p_method":[1/4, 1/4, 1/4, 1/4], ;
    metoda: 'extern';   config -> 'extern_kw' ;\n"""
        return info

    def __unpackConfigs(self):
        self.__fn      = {}
        self.__methods = {}
        self.__externs_fn = {}
        for idx, key in enumerate(self.__genome.keys(), 0):
            method = self.__chromosom_configs[key].pop("method", None)
            self.__methods[key] = method
            self.__extern_fn    = self.__chromosom_configs[key].pop("extern_fn", None)
            self.__fn[key]      = self.__unpackMethod(method, self.__extern_fn)

    def setParameters(self, **kw):
        super().setParameters(**kw)
        if (self.__extern_fn is not None):
            self.__extern_fn.setParameters(**kw)

    def crossoverAbstract(self, parent1, parent2, low, high):
        error_mesage = ""
        for chrom_name in self.__genome.keys():
            error_mesage += "Lipseste metoda '{}' pentru chromozomul '{}', functia de 'Crossover': config '{}'\n".format(self.__methods[chrom_name], chrom_name, self.__chromosom_configs[chrom_name])
        raise NameError(error_mesage)

    def crossoverChromosome(self):
        pass

    def crossoverGenome(self):
        pass

    def crossoverMixt(self):
        pass
