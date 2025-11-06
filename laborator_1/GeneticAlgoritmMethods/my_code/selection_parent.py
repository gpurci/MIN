#!/usr/bin/python

import numpy as np

class SelectionParent(RootGA):
    """
    Clasa 'Selection', ofera doar metode pentru a selecta unul din parinti in calitate de parinte 1 si 2
    Functia 'selectParent' are 2 parametri, fitness_value, fitness_partener.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self, config):
        self.setConfig(config)

    def __call__(self):
        return self.fn()

    def __config_fn(self):
        self.fn = self.selectParentAbstract
        if (self.__config is not None):
            if   (self.__config == "choise"):
                self.fn = self.selectParentChoice
            elif (self.__config == "roata"):
                self.fn = self.selectParentWheel
            elif (self.__config == "turneu"):
                self.fn = self.selectParentTour
        else:
            pass

    def setConfig(self, config):
        self.__config = config
        self.__config_fn()

    def startEpoch(self, fitness_values):
        total_fitness = fitness_values.sum()
        if (total_fitness != 0):
            self.fitness_values = fitness_values / total_fitness
        else:
            size = fitness_values.shape[0]
            self.fitness_values = np.full(fitness_values.shape[0], 1./size, dtype=np.float32)

    def selectParentAbstract(self, parent1, parent2, offspring):
        raise NameError("Lipseste configuratia pentru functia de 'SelectionParent': config '{}'".format(self.__config))

    def selectParentChoice(self):
        """Selecteaza un parinte aleator din populatie,
            - unde valoarea fitness este probabilitatea de a fi ales
        """
        # selectare aleatorie
        arg = np.random.choice(Selection.POPULATION_SIZE, size=None, p=self.fitness_values)
        return arg

    def selectParentWheel(self):
        """Selecteaza un parinte aleator din populatie,
            - unde valoarea fitness este probabilitatea de a fi ales
        """
        # selectie dupa compatibilitate, roata norocului
        current = 0
        # suma fitnesului asteptata
        pick    = np.random.uniform(low=0, high=1, size=None)
        # roata norocului
        for arg, fitness_value in enumerate(self.fitness_values, 0):
            current += fitness_value
            if (current > pick):
                break
        return arg

    def selectParentTour(self):
        """Selecteaza un parinte aleator din populatie,
            - unde valoarea fitness este probabilitatea de a fi ales
        """
        # selectie dupa compatibilitate, turneu
        args_k_tour = np.random.randint(low=0, high=Selection.POPULATION_SIZE, size=7)
        arg         = np.argmax(self.fitness_values[args_k_tour])
        return args_k_tour[arg]
