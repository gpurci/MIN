#!/usr/bin/python

import pandas as pd

class TestCallback(object):
    """
    Salveaza rezultatele obtinute pentru fiecare generatie in 'csv' file.
    """

    def __init__(self, filename):
        super().__init__(filename)

    def test(self):
        for generation in range(20):
            # call with positional arguments
            self(generation,
                 {"log_test_0": generation,
                  "log_test_2": 0})
        print("Test finished — check file:", self.filename)
