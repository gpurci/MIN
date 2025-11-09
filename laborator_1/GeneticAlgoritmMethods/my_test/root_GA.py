#!/usr/bin/python
import numpy as np
from my_code.root_GA import *

class TestRootGA(RootGA):

    def __init__(self, **config):
        super().__init__()

    def __permutePopulation(self, individ):
        new_individ = np.random.permutation(individ)

        start = np.array([self.GENOME_LENGTH-1], dtype=np.int32)
        end   = np.array([self.GENOME_LENGTH-1], dtype=np.int32)

        new_individ = np.concatenate((start, new_individ, end))
        return new_individ

    def initPopulation(self, population_size=-1):
        if population_size == -1:
            population_size = self.POPULATION_SIZE

        size = (population_size, self.GENOME_LENGTH-1)

        population = np.arange(np.prod(size), dtype=np.int32).reshape(*size)%(self.GENOME_LENGTH-1)
        population = np.apply_along_axis(self.__permutePopulation, axis=1, arr=population)

        return self.close_population(population)

    def close_population(self, pop):
        # if already closed, do nothing
        if pop.shape[1] == self.GENOME_LENGTH + 1:
            return pop
        return np.hstack([pop, pop[:, 0:1]])

