#!/usr/bin/python

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from laborator_1.GeneticAlgoritmMethods.my_code.callback import Callback

class TestCallback(Callback):
    def __init__(self, filename):
        super().__init__(filename)

    def test(self):
        for generation in range(20):
            # call with positional arguments
            self(generation,
                 {"log_test_0": generation,
                  "log_test_2": 0})
        print("Test finished — check file:", self.filename)
