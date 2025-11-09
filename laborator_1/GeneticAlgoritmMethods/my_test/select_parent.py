#!/usr/bin/python

import numpy as np
from my_code.select_parent import SelectParent
from my_test.root_GA         import TestRootGA


class TestSelectParent(TestRootGA):


    def __init__(self):
        super().__init__()

        self.selector = SelectParent("choice")
        self.selector.setParameters(
            POPULATION_SIZE = 4,
            SELECT_RATE     = 0.0
        )

        self.fitness = np.array([0.1, 0.2, 0.5, 0.2], dtype=np.float32)

    # helper
    def _run(self, config_name):
        print(f"\n=== {config_name} ===")
        self.selector.setConfig(config_name)
        print("DEBUG POPULATION_SIZE =", self.selector.POPULATION_SIZE)
        # trebuie înainte de selecție!
        self.selector.startEpoch(self.fitness.copy())

        picks = [int(self.selector()) for _ in range(20)]
        print("picks:", picks)

    # teste individuale
    def test_choice(self):         self._run("choice")
    def test_roata(self):          self._run("roata")
    def test_turneu(self):         self._run("turneu")
    def test_turneu_choice(self):  self._run("turneu_choice")
    def test_crescator(self):      self._run("crescator")
    def test_mixt(self):           self._run("mixt")
