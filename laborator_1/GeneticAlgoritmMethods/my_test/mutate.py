#!/usr/bin/python

import numpy as np
from my_code.mutate import *
from my_test.root_GA import *

class TestMutate(Mutate, TestRootGA):
    """
    Clasa 'mutate', ofera doar metode pentru a face mutatia genetica a unui individ din populatie.
    Functia 'mutate' are 3 parametri, parinte1, parinte2, descendent.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self, config):
        super().__init__(config)

    def test(self, config):
        self.setParameters(
            MUTATION_RATE = 0.9,  # threshold-ul pentru a face o mutatie genetica
            GENOME_LENGTH = 20
            )
        self.setConfig(config)
        population = self.initPopulation(3)
        parent1 = population[0]
        parent2 = population[1]
        offspring = population[2]
        print("parent1 {}".format(parent1))
        print("parent2 {}".format(parent2))
        print("offspring {}".format(offspring))

        for _ in range(10):
            mutate_offspring = self(parent1, parent2, offspring.copy())
            tmp_cmp = mutate_offspring!=offspring
            if (tmp_cmp.sum() != 0):
                print("Operatia de mutatie a fost aplicata: {}".format(tmp_cmp))
            else:
                print("Operatia de mutatie 'lipseste'")
