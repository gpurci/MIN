#!/usr/bin/python

import numpy as np
from my_code.crossover import Crossover
from my_test.root_GA   import TestRootGA

class TestCrossover(TestRootGA):

    def __init__(self):
        super().__init__()

        # forțăm rate 100% pentru a vedea efectul la fiecare apel
        self.setParameters(GENOME_LENGTH=10, CROSSOVER_RATE=1.0)

        # p1/p2 diferite + un bloc comun [3:8] (ca să declanșeze perm_sim/flip_sim)
        pop = self.initPopulation(3)
        self.p1 = pop[0].copy()
        self.p2 = np.roll(self.p1, 1).copy()     # p2 clar diferit

        block = np.random.randint(0, 10, size=5)
        self.p1[3:8] = block
        self.p2[3:8] = block

        # obiectul de test
        self.c = Crossover("diff")
        self.c.setParameters(GENOME_LENGTH=10, CROSSOVER_RATE=1.0)

    def _run(self, config_name, max_tries=20):
        self.c.setConfig(config_name)

        print(f"\n=== {config_name} ===")
        print("parent1:", self.p1)
        print("parent2:", self.p2)

        # încercăm de câteva ori (pentru că start/end sau permutările sunt random)
        for attempt in range(1, max_tries+1):
            before = self.p1.copy()
            after  = self.c(self.p1, self.p2)
            diff   = np.where(before != after)[0]
            if diff.size > 0:
                print(f"attempt {attempt}:")
                print("before:", before)
                print("after :", after)
                print("diff idx:", diff)
                break
        else:
            # dacă nu s-a schimbat nimic în max_tries încercări, raportează
            print(f"(no visible change after {max_tries} tries)")
            print("last before:", before)
            print("last after :", after)

    def test_diff(self):       self._run("diff")
    def test_split(self):      self._run("split")
    def test_perm_sim(self):   self._run("perm_sim")
    def test_flip_sim(self):   self._run("flip_sim")
    def test_mixt(self):       self._run("mixt")
