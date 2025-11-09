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
    def __init__(self, config):
        super().__init__()
        self.setConfig(config)

    def __call__(self, parent1, parent2):
        return self.crossover(parent1, parent2)

    def __config_fn(self):
        self.crossover = self.crossoverAbstract
        if self.__config is not None:
            if self.__config == "diff":
                self.crossover = self.crossoverDiff
            elif self.__config == "split":
                self.crossover = self.crossoverSplit
        else:
            pass

    def help(self):
        info = """Crossover: 
        metode de config: 'diff', 'split'\n"""
        return info

    def setConfig(self, config):
        self.__config = config
        self.__config_fn()

    def crossoverAbstract(self, parent1, parent2):
        raise NameError("Lipseste configuratia pentru functia de 'Crossover': config '{}'".format(self.__config))

    def crossoverDiff(self, parent1, parent2):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        # mosteneste parinte1
        offspring = parent1.copy()
        # modifica doar genele care sunt diferite
        mask = parent1!=parent2
        diff_locus = np.argwhere(mask)
        size = diff_locus.shape[0]
        if (size >= 4):
            tmp_size   = np.random.randint(low=size//3, high=size//2, size=None)
            diff_locus = diff_locus.reshape(-1)
            diff_locus = np.random.permutation(diff_locus)[:tmp_size]
            offspring[diff_locus] = parent2[diff_locus]
        return offspring

    def crossoverSplit(self, parent1, parent2):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        nbr_childs - cati copii vor fi generati de acesti 2 parinti
        """
        # mosteneste parinte1
        offspring = parent1.copy()
        # selectarea diapazonului de mostenire
        start, end = np.random.randint(low=1, high=self.GENOME_LENGTH, size=2)
        # corectie diapazon
        if start > end: start, end = end, start
        # copierea rutei din cel de al doilea parinte
        offspring[start:end] = parent2[start:end]
        return offspring
