#!/usr/bin/python

import numpy as np
from root_GA import *

class Mutate(RootGA):
    """
    Clasa 'Mutate', ofera doar metode pentru a face mutatia genetica a unui individ din populatie.
    Functia 'mutate' are 3 parametri, parinte1, parinte2, descendent.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, genome, method, **configs):
        super().__init__()
        self.__genome  = genome
        self.__configs = configs
        self.__unpackConfigs(method)

    def __call__(self, parent1, parent2, offspring, **kw):
        return self.__man_fn(parent1, parent2, offspring, **kw)

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
        info = "Mutate: method '{}'\n".format(self.__method)
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
        fn = self.mutateChromosomeAbstract
        if (extern_fn is not None):
            fn = extern_fn
        return fn

    def __unpackManMethod(self, method):
        fn = self.mutateAbstract
        if (method is not None):
            if   (method == "chromosome"):
                fn = self.mutateChromosome
            elif (method == "genome"):
                fn = self.mutateGenome
            elif (method == "mixt"):
                fn = self.mutateMixt
        return fn

    def help(self):
        info = """Mutate:
    metoda: 'chromosome'; config None;
    metoda: 'genome';     config None;
    metoda: 'mixt';       config -> "p_select":[1/2, 1/2], ;\n"""
        return info

    def __unpackChromosomeConfigs(self, method):
        # unpack cromosome method
        chromosome_fn = {}
        if (method == "chromosome"):
            for key in self.__genome.keys():
                extern_fn = self.__configs.get(key, None)
                fn        = self.__unpackChromosomeMethod(extern_fn)
                chromosome_fn[key] = fn
        return chromosome_fn

    def __unpackGenomeConfigs(self, method):
        chromosome_fn = {}
        if (method == "genome"):
            extern_fn = self.__configs.get("genome", None)
            fn        = self.__unpackChromosomeMethod(extern_fn)
            self.__chromosome_fn["genome"] = fn
        return chromosome_fn

    def __unpackMixtConfigs(self, method):
        chromosome_fn = {}
        if (method == "mixt"):
            chromosome_fn = self.__unpackChromosomeConfigs("chromosome")
            tmp_fn        = self.__unpackChromosomeConfigs("genome")
            chromosome_fn.update(tmp_fn)
        return chromosome_fn

    def __unpackConfigs(self, method):
        # unpack manager method
        self.__method = method
        self.__man_fn = self.__unpackManMethod(self.__method)
        # unpack cromosome method
        tmp_chr_fn = self.__unpackChromosomeConfigs(self.__method)
        tmp_gen_fn = self.__unpackGenomeConfigs(self.__method)
        tmp_mxt_fn = self.__unpackMixtConfigs(self.__method)
        self.__chromosome_fn = tmp_chr_fn
        self.__chromosome_fn.update(tmp_gen_fn)
        self.__chromosome_fn.update(tmp_mxt_fn)

    def setParameters(self, **kw):
        super().setParameters(**kw)
        for key in self.__chromosome_fn.keys():
            obj_chromosome = self.__chromosome_fn[key]
            if ((obj_chromosome is not None) and (hasattr(obj_chromosome, "setParameters"))):
                obj_chromosome.setParameters(**kw)

    def mutateAbstract(self, *args, **kw):
        error_mesage = "Functia 'Mutate', lipseste metoda '{}', config: '{}'\n".format(self.__method, self.__configs)
        raise NameError(error_mesage)

    def mutateChromosomeAbstract(self, *args, **kw):
        error_mesage = "Functia 'Mutate', lipseste metoda '{}', chromozomul '{}', config: '{}'\n".format(self.__method, self.__chromosome_name, self.__configs)
        raise NameError(error_mesage)

    def __mutateChromosome(self, parent1, parent2, offspring, chrom_name, **kw):
        # calcularea ratei de probabilitate a mutatiei
        rate = np.random.uniform(low=0, high=1, size=None)
        if (rate <= self.MUTATION_RATE): # aplicarea operatiei de mutatie
            offspring = self.__chromosome_fn[chrom_name](
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
        for self.__chromosome_name in self.__genome.keys():
            chromosome_val = self.__mutateChromosome(parent1, parent2, offspring, self.__chromosome_name, **kw)
            tmp_genome.append(chromosome_val)
        return self.__genome.concat(tmp_genome)

    def mutateGenome(self, parent1, parent2, offspring, **kw):
        # adaugare mutate rate
        rate = np.random.uniform(low=0, high=1, size=None)
        if (rate <= self.CROSSOVER_RATE): # operatia de incrucisare
            offspring = self.__chromosome_fn["genome"](parent1, parent2, offspring, **kw)
        else: # selectie chromosom intreg
            pass
            #offspring = offspring
        return offspring

    def mutateMixt(self, parent1, parent2, offspring, **kw):
        error_mesage = "Lipseste metoda 'mutateMixt' functia de 'Mutate': config '{}'\n".format(self.__configs)
        raise NameError(error_mesage)
