#!/usr/bin/python

import numpy as np
from my_code.root_GA import *

class Crossover(RootGA):
    """
    Clasa 'Crossover', ofera doar metode pentru a face incrucisarea genetica a doi parinti
    Functia 'crossover' are 2 parametri, parinte1, parinte2.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self, config):
        self.__config = config
        self.__config_fn()

    def __config_fn(self):
        self.crossover = self.crossoverAbstract
        if (self.__config is not None):
            if   (self.__config == "test"):
                self.crossover = self.testParentClass
            elif (self.__config == "vecin"):
                self.crossover = self.crossoverNeighbors
            elif (self.__config == "swap"):
                self.crossover = self.crossoverSwap
        else:
            pass

    def crossoverAbstract(self, parent1, parent2):
        raise NameError("Configuratie gresita pentru functia de 'Crossover'")

    def crossover(self, parent1, parent2):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        # mosteneste parinte1
        offspring = parent1.copy()
        # modifica doar genele care sunt diferite
        mask = parent1!=parent2
        mask[[0, -1]] = False # pastreaza orasul de start
        diff_locus = np.argwhere(mask)
        size = diff_locus.shape[0]
        if (size > 4):
            tmp_size = np.random.randint(low=size//3, high=size//2, size=None)
            diff_locus = diff_locus.reshape(-1)
            diff_locus = np.random.permutation(diff_locus)[:tmp_size]
            child[diff_locus] = parent2[diff_locus]
        else:
            start, end = np.random.randint(low=1, high=self.GENOME_LENGTH, size=2)
            if (start > end): start, end = end, start
            # copierea rutei din parintele 2
            child[start:end] = parent2[start:end]

        return child


    def crossoverIndivid(self, parent1, parent2, arg, childs, coords, crossover_conds):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        arg     - pozitia din vector
        childs  - vector de indivizi, pentru copii
        coords  - vector de coordonate start, end
        crossover_conds - vector pentru metodele de aplicare a incrucisarii
        """
        # creare un copil fara mostenire
        child       = childs[arg]
        # selectarea diapazonului de mostenire
        start, end  = coords[arg]
        cond = crossover_conds[arg]
        #print("coords ", coords[arg], "cond", cond)
        if (start > end):
            start, end = end, start
        if (cond == 0):
            # copierea rutei din primul parinte
            child[start:end] = parent1[start:end]
            # copierea rutei din cel de al doilea parinte
            child[:start] = parent2[:start]
            child[end:]   = parent2[end:]
        elif (cond == 1):
            # modifica doar genele care sunt diferite
            mask = parent1!=parent2
            mask[[0, -1]] = False # pastreaza orasul de start
            args = np.argwhere(mask).reshape(-1)
            if (args.shape[0] > 1):
                args = args.reshape(-1)
                tmp_size = min(end-start, args.shape[0]//2)
                args = np.random.choice(args, size=tmp_size)
                child[:]    = parent1[:]
                child[args] = parent2[args]

    def crossover(self, parent1, parent2, nbr_childs=1):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        nbr_childs - cati copii vor fi generati de acesti 2 parinti
        """
        # creare un copil fara mostenire
        childs = np.zeros((nbr_childs, TSP.GENOME_LENGTH+1), dtype=np.int32)
        # selectarea diapazonului de mostenire
        coords = np.random.randint(low=1, high=TSP.GENOME_LENGTH+1, size=(nbr_childs, 2))
        # medodele de aplicare a incrucisarii
        # cond 0 -> selectare o zona aleatorie de gene
        #      1 -> se face incrucisare doar la genele diferite
        crossover_conds = np.random.choice([0, 1], size=nbr_childs, p=[0.6, 0.4])
        for arg in range(nbr_childs):
            self.crossoverIndivid(parent1, parent2, arg, childs, coords, crossover_conds)
        return childs
