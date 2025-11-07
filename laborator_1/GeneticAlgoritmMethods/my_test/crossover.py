#!/usr/bin/python

import numpy as np
from my_code.crossover import *
from my_test.root_GA import *

class TestCrossover(Crossover, TestRootGA):
    """
    Clasa 'Crossover', ofera doar metode pentru a face incrucisarea genetica a doi parinti
    Functia 'crossover' are 2 parametri, parinte1, parinte2.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self, config):
        super().__init__(config)

    def test(self, config):
        self.setConfig(config)
        population = self.initPopulation(2)
        parent1 = population[0]
        parent2 = population[1]
        print("parent1 {}".format(parent1))
        print("parent2 {}".format(parent2))

        for _ in range(10):
            offspring = self(parent1, parent2)
            tmp_cmp = parent1!=offspring
            if (tmp_cmp.sum() != 0):
                print("Operatia de incrucisare a fost aplicata: {}".format(tmp_cmp))
            else:
                print("Operatia de incrucisare 'lipseste'")

        # TEST: first and last gene must remain same as parent1
        print("\nStart/End preservation check:")
        for _ in range(5):
            off = self(parent1, parent2)
            if off[0] != parent1[0] or off[-1] != parent1[-1]:
                print("ERROR start/end changed!!  -> ", off)
            else:
                print("OK ", off[0], off[-1])

        # TEST: also run split config
        print("\nTesting SPLIT crossover:")
        self.setConfig("split")
        for _ in range(5):
            off = self(parent1, parent2)
            print("offspring_split = ", off)

