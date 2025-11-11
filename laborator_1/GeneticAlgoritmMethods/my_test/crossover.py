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
        TestRootGA.__init__(self)
        Crossover.__init__(self, config)

        self.GENOME_LENGTH   = 8
        self.POPULATION_SIZE = 2

    def test(self, config):
        self.setConfig(config)

        population = self.initPopulation(2)
        parent1 = population[0]
        parent2 = population[1]

        print("parent1", parent1)
        print("parent2", parent2)

        for _ in range(10):
            offspring = self(parent1, parent2)
            diff = parent1!=offspring
            if diff.sum() != 0:
                print("OK - crossover applied:", offspring)
            else:
                print("NO crossover (offspring identical)")

        print("\nTesting SPLIT crossover:")
        self.setConfig("split")
        for _ in range(5):
            off = self(parent1, parent2)
            print("offspring split:", off)
