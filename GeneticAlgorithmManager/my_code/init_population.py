#!/usr/bin/python

import numpy as np
from extern_fn import *

class InitPopulation():
    """
    Clasa 'InitPopulation', ofera doar metode pentru a initializa populatia.
    Functia 'initPopulation' are 1 parametru, numarul populatiei.
    Metoda '__config_fn', selecteaza functia de initializare.
    Metoda '__call__', aplica functia de initializare ce a fost selectata in '__config_fn'
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self, extern_fn=None):
        super().__init__(extern_fn, "InitPopulation")

    def __call__(self, size, genoms):
        return self.__extern_fn(size, genoms)
