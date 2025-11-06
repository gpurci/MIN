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

    def __init__(self, **config):
        super().__init__(**config)

    def __permutePopulation(self, individ):
        new_individ = np.random.permutation(individ)
        new_individ = np.concatenate((RootGA.GENOME_LENGTH-1, new_individ, RootGA.GENOME_LENGTH-1), axis=None)
        return new_individ

    def initPopulation(self, population_size=-1):
        """Initializarea populatiei, cu drumuri aleatorii"""
        if (population_size == -1):
            population_size = RootGA.POPULATION_SIZE
        size = (population_size, RootGA.GENOME_LENGTH-1)
        population = np.arange(np.prod(size), dtype=np.int32).reshape(*size)%(RootGA.GENOME_LENGTH-1)
        population = np.apply_along_axis(self.__permutePopulation, axis=1, arr=population)
        return population
