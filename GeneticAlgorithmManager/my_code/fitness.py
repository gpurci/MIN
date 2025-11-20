#!/usr/bin/python

import numpy as np
from extern_fn import *

class Fitness():
    """
    Clasa 'Fitness', ofera doar metode pentru a calcula functia fitness din populatie.
    Functia 'fitness' are 1 parametru, numarul populatiei.
    Metoda '__call__', aplica functia de initializare ce a fost selectata in '__config_fn'
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, extern_fn=None):
        super().__init__(extern_fn, "Fitness")

    def __call__(self, metric_values):
        return self.__extern_fn(metric_values)
