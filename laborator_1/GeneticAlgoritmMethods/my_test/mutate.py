#!/usr/bin/python
import sys
from pathlib import Path

import numpy as np
from my_code.mutate import *
from my_test.root_GA import *

class TestMutate(Mutate, TestRootGA):

    def __init__(self, config):
        TestRootGA.__init__(self)
        Mutate.__init__(self, config)

    def test(self, config):
        self.setParameters(
            MUTATION_RATE = 0.9,
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
            if tmp_cmp.sum() != 0:
                print("Operatia de mutatie a fost aplicata:", mutate_offspring)
            else:
                print("Operatia de mutatie 'lipseste'")
