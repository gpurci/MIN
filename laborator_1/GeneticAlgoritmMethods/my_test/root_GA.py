#!/usr/bin/python

import numpy as np
from my_code.root_GA import *

class TestRootGA(RootGA):
    """
    Clasa root pentru algoritmi genetici:
    In cadrul clasei root:
        - initializare variabile generale, pentru rularea algoritmului genetic
        - setare variabile generale
        - scurta descriere
    """

    def __init__(self):
        super().__init__()

    def __permutePopulation(self, individ):
        new_individ = np.random.permutation(individ)
        return new_individ

    def initPopulation(self, population_size=-1):
        if population_size == -1:
            population_size = self.POPULATION_SIZE

        size = (population_size, self.GENOME_LENGTH)

        population = np.arange(np.prod(size), dtype=np.int32).reshape(*size) % self.GENOME_LENGTH
        population = np.apply_along_axis(self.__permutePopulation, axis=1, arr=population)

        return population
