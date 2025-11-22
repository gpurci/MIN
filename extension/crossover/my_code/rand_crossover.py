#!/usr/bin/python

import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import *

class CrossoverOX(RootGA):
    """
    Clasa 'CrossoverOX', ofera doar metode pentru a face incrucisarea genetica a doi parinti
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
        info  = "CrossoverOX: method '{}'\n".format(self.__method)
        tmp   = "configs: '{}'\n".format(self.__configs)
        info += "\t{}".format(tmp)
        return info

    def __unpackMethod(self, method):
        fn = self.crossoverAbstract
        if (method is not None):
            if   (method == "scramble"):
                fn = self.crossoverScramble
            elif (method == "inversion"):
                fn = self.crossoverInversion
            elif (method == "inversion"):
                fn = self.crossoverRoll
            elif (method == "mixt"):
                fn = self.crossoverMixt

        return fn

    def help(self):
        info = """CrossoverOX:
    metoda: 'diff';     config None;
    metoda: 'split';    config None;
    metoda: 'perm_sim'; config None;
    metoda: 'flip_sim'; config None;
    metoda: 'mixt';     config -> "p_select":[1/4, 1/4, 1/4, 1/4], ;\n"""
        print(info)

    def crossoverAbstract(self, parent1, parent2):
        error_mesage = "Functia 'CrossoverOX', lipseste metoda '{}', config: '{}'\n".format(self.__method, self.__configs)
        raise NameError(error_mesage)

    def crossoverScramble(self, parent1, parent2):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        # mosteneste parinte1
        offspring = parent1.copy()
        # selectarea diapazonului de mostenire
        locus = np.random.randint(low=0, high=self.GENOME_LENGTH, size=self.GENOME_LENGTH//2)
        genes_p1 = parent1[locus]
        offspring[locus] = np.random.permutation(genes_p1)
        return offspring

    def crossoverInversion(self, parent1, parent2):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        # mosteneste parinte1
        offspring = parent1.copy()
        # selectarea diapazonului de mostenire
        locus = np.random.randint(low=0, high=self.GENOME_LENGTH, size=self.GENOME_LENGTH//2)
        genes_p1 = parent1[locus]
        offspring[locus] = np.flip(genes_p1)
        return offspring

    def crossoverRoll(self, parent1, parent2):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        # mosteneste parinte1
        offspring = parent1.copy()
        # valoarea shiftarii
        size_shift = np.random.randint(low=1, high=self.GENOME_LENGTH//2-1, size=None)
        # selectarea diapazonului de mostenire
        locus = np.random.randint(low=0, high=self.GENOME_LENGTH, size=self.GENOME_LENGTH//2)
        offspring[locus] = np.roll(genes_p1, size_shift)
        return offspring

    def crossoverMixt(self, parent1, parent2, p_select=None):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        cond = np.random.choice([0, 1, 2], size=None, p=p_select)
        if   (cond == 0):
            offspring = self.crossoverScramble(parent1, parent2)
        elif (cond == 1):
            offspring = self.crossoverInversion(parent1, parent2)
        elif (cond == 2):
            offspring = self.crossoverRoll(parent1, parent2)
        return offspring
