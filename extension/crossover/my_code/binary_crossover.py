#!/usr/bin/python

import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import *

class CrossoverBinary(RootGA):
    """
    Clasa 'CrossoverBinary', ofera doar metode pentru a face incrucisarea genetica a doi parinti
    Functia 'crossover' are 2 parametri, parinte1, parinte2.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, **configs):
        super().__init__()
        self.__method  = method
        self.__configs = configs
        self.__fn = self.__unpackMethod(method)

    def __call__(self, parent1, parent2):
        return self.__fn(parent1, parent2, **self.__configs)

    def __str__(self):
        info  = "CrossoverBinary: method '{}'\n".format(self.__method)
        tmp   = "configs: '{}'\n".format(self.__configs)
        info += "\t{}".format(tmp)
        return info

    def __unpackMethod(self, method):
        fn = self.crossoverAbstract
        if (method is not None):
            if   (method == "single_point"):
                fn = self.crossoverSP
            elif (method == "two_point"):
                fn = self.crossoverTwoP
            elif (method == "uniform"):
                fn = self.crossoverUniform
            elif (method == "mixt"):
                fn = self.crossoverMixt

        return fn

    def help(self):
        info = """CrossoverBinary:
    metoda: 'single_point'; config None;
    metoda: 'two_point';    config None;
    metoda: 'uniform';      config None;
    metoda: 'mixt';         config -> "p_select":[1/3, 1/3, 1/3], ;\n"""
        print(info)

    def crossoverAbstract(self, parent1, parent2):
        error_mesage = "Functia 'CrossoverBinary', lipseste metoda '{}', config: '{}'\n".format(self.__method, self.__configs)
        raise NameError(error_mesage)

    def crossoverSP(self, parent1, parent2):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        # mosteneste parinte1
        offspring = parent1.copy()
        # selectarea diapazonului de mostenire
        sp = np.random.randint(low=self.GENOME_LENGTH//4, high=3*self.GENOME_LENGTH//4, size=None)
        # copierea rutei din cel de al doilea parinte
        offspring[sp:] = parent2[sp:]
        return offspring

    def crossoverTwoP(self, parent1, parent2):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        # mosteneste parinte1
        offspring = parent1.copy()
        # selectarea diapazonului de mostenire
        start, end = np.random.randint(low=1, high=self.GENOME_LENGTH-1, size=2)
        # corectie diapazon
        if (start > end): start, end = end, start
        # copierea rutei din cel de al doilea parinte
        offspring[start:end] = parent2[start:end]
        return offspring

    def crossoverUniform(self, parent1, parent2):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        # mosteneste parinte1
        offspring = parent1.copy()
        # selectarea diapazonului de mostenire
        locus = np.random.randint(low=0, high=self.GENOME_LENGTH, size=self.GENOME_LENGTH//2)
        offspring[locus] = parent2[locus]
        return offspring

    def crossoverMixt(self, parent1, parent2, p_select=None):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        cond = np.random.choice([0, 1, 2], size=None, p=p_select)
        if   (cond == 0):
            offspring = self.crossoverSP(parent1, parent2)
        elif (cond == 1):
            offspring = self.crossoverTwoP(parent1, parent2)
        elif (cond == 2):
            offspring = self.crossoverUniform(parent1, parent2)
        return offspring
