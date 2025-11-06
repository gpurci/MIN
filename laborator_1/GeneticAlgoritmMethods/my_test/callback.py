#!/usr/bin/python

import pandas as pd

class TestCallback(object):
    """
    Salveaza rezultatele obtinute pentru fiecare generatie in 'csv' file.
    """

    def __init__(self, filename):
        super().__init__(filename)

    def test(self, epoch, logs):
        for i in range(20):
            self(1, {"log_test_0":i, "log_test_2":0})

