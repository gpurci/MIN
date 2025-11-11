#!/usr/bin/python

from my_code.mutate   import Mutate
from my_test.root_GA import TestRootGA


class TestMutate(TestRootGA):

    def __init__(self, config):
        super().__init__()

        self.setParameters(
            GENOME_LENGTH = 10,
            MUTATION_RATE = 1.0  # fortam mutatia
        )

        # instanta Mutate reala
        self.m = Mutate(config)
        self.m.setParameters(GENOME_LENGTH=10, MUTATION_RATE=1.0)

        pop = self.initPopulation(3)
        self.pop = pop
        self.child = pop[2].copy()

    # ================= helper ======================

    def _parents(self, config_name):
        sim_modes = ["perm_sim", "roll_sim", "flip_sim"]
        if config_name in sim_modes:
            return self.pop[0].copy(), self.pop[0].copy()   # identici
        return self.pop[0].copy(), self.pop[1].copy()       # diferiti

    def _run(self, config_name):
        self.m.setConfig(config_name)

        p1, p2 = self._parents(config_name)
        before = self.child.copy()
        after  = self.m(p1, p2, before.copy())
        diff   = self.m._diff(before, after)

        print(f"\n=== {config_name} ===")
        print("p1:", p1)
        print("p2:", p2)
        print("before:", before)
        print("after :", after)
        print("diff :", diff)

    # ================= tests =======================

    def test_swap(self):       self._run("swap")
    def test_scramble(self):   self._run("scramble")
    def test_inversion(self):  self._run("inversion")
    def test_roll(self):       self._run("roll")
    def test_insertion(self):  self._run("insertion")
    def test_diff_swap(self):  self._run("diff_swap")
    def test_roll_sim(self):   self._run("roll_sim")
    def test_perm_sim(self):   self._run("perm_sim")
    def test_flip_sim(self):   self._run("flip_sim")
    def test_mixt(self):       self._run("mixt")
