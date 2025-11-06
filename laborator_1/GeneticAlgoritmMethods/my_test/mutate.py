#!/usr/bin/python

import numpy as np
from my_code.mutate import *

class TestMutate(Mutate):
    """
    Clasa 'mutate', ofera doar metode pentru a face mutatia genetica a unui individ din populatie.
    Functia 'mutate' are 3 parametri, parinte1, parinte2, descendent.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self, **config):
        super().__init__(**config)

    def __permutePopulation(self, individ):
        new_individ = np.random.permutation(individ)
        new_individ = np.concatenate((Mutate.GENOME_LENGTH-1, new_individ, Mutate.GENOME_LENGTH-1), axis=None)
        return new_individ

    def initPopulation(self, population_size=-1):
        """Initializarea populatiei, cu drumuri aleatorii"""
        if (population_size == -1):
            population_size = Mutate.POPULATION_SIZE
        size = (population_size, Mutate.GENOME_LENGTH-1)
        population = np.arange(np.prod(size), dtype=np.int32).reshape(*size)%(Mutate.GENOME_LENGTH-1)
        population = np.apply_along_axis(self.__permutePopulation, axis=1, arr=population)
        return population

    def __call__(self, config):
        self.setParameters(
            MUTATION_RATE = 0.9,  # threshold-ul pentru a face o mutatie genetica
            GENOME_LENGTH = 20
            )
        population = self.initPopulation(3)
        parent1 = population[0]
        parent2 = population[1]
        offspring = population[0]
        print("parent1 {}".format(parent1))
        print("parent2 {}".format(parent2))
        print("offspring {}".format(offspring))

        for _ in range(10):
            mutate_offspring = self.mutate(parent1, parent2, offspring.copy())
            tmp_cmp = mutate_offspring!=offspring
            if (tmp_cmp.sum() != 0):
                print("Operatia de mutatie a fost aplicata: {}".format(tmp_cmp))
            else:
                print("Operatia de mutatie 'lipseste'")
