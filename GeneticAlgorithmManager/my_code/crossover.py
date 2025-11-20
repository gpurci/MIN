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
    def __init__(self, genome, **configs):
        super().__init__()
        self.__genome  = genome
        self.__configs = configs
        self.__unpackConfigs()

    def __call__(self, parent1, parent2, **kw):
        return self.__man_fn(parent1, parent2, **kw)

    def __str_chromosome(self):
        info = ""
        for chrom_name in self.__genome.keys():
            obj_chromosome = self.__chromosome_fn[chrom_name]
            if (obj_chromosome is not None):
                tmp = "Chromozome name: '{}', info :'{}'\n".format(chrom_name, str(obj_chromosome))
            else:
                tmp = "Chromozome name: '{}', info :'{}'\n".format(chrom_name, None)
            info += "\t{}".format(tmp)
        return info

    def __str_genome(self):
        info = ""
        obj_chromosome = self.__chromosome_fn["genome"]
        if (obj_chromosome is not None):
            tmp = "Genome info: '{}'\n".format(str(obj_chromosome))
        else:
            tmp = "Genome info: '{}'\n".format(None)
        info += "\t{}".format(tmp)
        return info

    def __str__(self):
        info = "Crossover: method '{}'\n".format(self.__method)
        if   (self.__method == "chromosome"):
            info += self.__str_chromosome()
        elif (self.__method == "genome"):
            info += self.__str_genome()
        elif (self.__method == "mixt"):
            info += "\tp_select: {}".format(self.__configs.get("p_select", None))
            info += self.__str_chromosome()
            info += self.__str_genome()
        return info

    def __unpackChromosomeMethod(self, extern_fn):
        fn = self.crossoverChromosomeAbstract
        if (extern_fn is not None):
            fn = extern_fn
        return fn

    def __unpackManMethod(self, method):
        fn = self.crossoverAbstract
        if (method is not None):
            if   (method == "chromosome"):
                fn = self.crossoverChromosome
            elif (method == "genome"):
                fn = self.crossoverGenome
            elif (method == "mixt"):
                fn = self.crossoverMixt
        return fn

    def help(self):
        info = """Crossover:
    metoda: 'chromosome'; config None;
    metoda: 'genome';     config None;
    metoda: 'mixt';       config -> "p_select":[1/2, 1/2], ;\n"""
        return info

    def __unpackConfigs(self):
        # unpack manager method
        self.__method = self.__configs.pop("method", None)
        self.__man_fn = self.__unpackManMethod(self.__method)
        # unpack cromosome method
        self.__chromosome_fn = {}
        for key in self.__genome.keys():
            extern_fn = self.__configs.get(key, None)
            fn        = self.__unpackChromosomeMethod(extern_fn)
            self.__chromosome_fn[key] = fn
        else:
            extern_fn = self.__configs.get("genome", None)
            fn        = self.__unpackChromosomeMethod(extern_fn)
            self.__chromosome_fn[key] = fn

    def setParameters(self, **kw):
        super().setParameters(**kw)
        for key in self.__chromosome_fn.keys():
            obj_chromosome = self.__chromosome_fn[key]
            if ((obj_chromosome is not None) and (hasattr(obj_chromosome, "setParameters"))):
                obj_chromosome.setParameters(**kw)

    def crossoverAbstract(self, *args, **kw):
        error_mesage = "Functia 'Crossover', lipseste metoda '{}', config: '{}'\n".format(self.__method, self.__configs)
        raise NameError(error_mesage)

    def crossoverChromosomeAbstract(self, *args, **kw):
        error_mesage = "Functia 'Crossover', lipseste metoda '{}', chromozomul '{}', config: '{}'\n".format(self.__method, self.__chromosome_name, self.__configs)
        raise NameError(error_mesage)

    def __crossoverChromosome(self, parent1, parent2, chrom_name, **kw):
        # adaugare crossover rate
        rate = np.random.uniform(low=0, high=1, size=None)
        if (rate <= self.CROSSOVER_RATE): # operatia de incrucisare
            offspring = self.__chromosome_fn[chrom_name](
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
        for self.__chromosome_name in self.__genome.keys():
            chromosome_val = self.__crossoverChromosome(parent1, parent2, self.__chromosome_name, **kw)
            tmp_genome.append(chromosome_val)
        return self.__genome.concat(tmp_genome)

    def crossoverGenome(self, parent1, parent2, **kw):
        # adaugare crossover rate
        rate = np.random.uniform(low=0, high=1, size=None)
        if (rate <= self.CROSSOVER_RATE): # operatia de incrucisare
            offspring = self.__chromosome_fn["genome"](parent1, parent2, **kw)
        else: # selectie chromosom intreg
            offspring = parent1.copy()
        return offspring

    def crossoverMixt(self, parent1, parent2, **kw):
        error_mesage = "Lipseste metoda 'crossoverMixt' functia de 'Crossover': config '{}'\n".format(self.__configs)
        raise NameError(error_mesage)
