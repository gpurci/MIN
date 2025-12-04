#!/usr/bin/python

import numpy as np
from extension.ga_base import *

class SelectParentBase(GABase):
    """
    Clasa 'SelectParentBase', 
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, name="SelectParentBase", **configs):
        super().__init__(method, name=name, **configs)
        self.fitness_values = None

    def setFitnessValues(self, fitness_values):
        self.fitness_values = fitness_values

    def selectParentChoice(self):
        """Selecteaza un parinte aleator din populatie,
            - unde valoarea fitness este probabilitatea de a fi ales
        """
        # selectare aleatorie
        arg = np.random.choice(self.POPULATION_SIZE, size=None, p=self.fitness_values)
        return arg

    def selectParentUniform(self):
        """Selecteaza un parinte aleator din populatie,
            - unde valoarea fitness este probabilitatea de a fi ales
        """
        # selectare aleatorie
        arg = np.random.randint(low=0, high=self.POPULATION_SIZE, size=None)
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

    def selectParentSortWheel(self):
        """Selecteaza un parinte aleator din populatie,
            - unde valoarea fitness este probabilitatea de a fi ales
        """
        # selectie dupa compatibilitate, roata norocului
        current = 0
        # suma fitnesului asteptata
        pick    = np.random.uniform(low=0, high=1, size=None)
        # roata norocului
        sort_args = np.argsort(self.fitness_values)
        for arg, fitness_value in enumerate(self.fitness_values[sort_args], 0):
            current += fitness_value
            if (current > pick):
                break
        return sort_args[arg]

    def selectParentTour(self, size_subset=7):
        """Selecteaza un parinte aleator din populatie,
            - unde valoarea fitness este probabilitatea de a fi ales
        """
        # selectie dupa compatibilitate, turneu
        args_k_tour = np.random.randint(low=0, high=self.POPULATION_SIZE, size=size_subset)
        arg         = np.argmax(self.fitness_values[args_k_tour])
        return args_k_tour[arg]

    def selectParentTourChoice(self, size_subset=7):
        """Selecteaza un parinte aleator din populatie,
            - unde valoarea fitness este probabilitatea de a fi ales
        """
        # selectie dupa compatibilitate, turneu
        args_k_tour = np.random.choice(self.POPULATION_SIZE, size=size_subset, p=self.fitness_values)
        arg         = np.argmax(self.fitness_values[args_k_tour])
        return args_k_tour[arg]

