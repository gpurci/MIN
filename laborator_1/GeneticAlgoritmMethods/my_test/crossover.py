#!/usr/bin/python

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from laborator_1.GeneticAlgoritmMethods.my_code.crossover import Crossover
from laborator_1.GeneticAlgoritmMethods.my_test.root_GA import TestRootGA

class TestCrossover(Crossover, TestRootGA):

    def __init__(self, config):
        TestRootGA.__init__(self)
        Crossover.__init__(self, config)

        self.GENOME_LENGTH   = 8
        self.POPULATION_SIZE = 2

    def test(self, config):
        self.setConfig(config)

        # build two random parents
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
