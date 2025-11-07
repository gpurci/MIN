#!/usr/bin/python

import numpy as np
from my_test.root_GA import *


class InitPopulation(RootGA):
    """
    Clasa 'InitPopulation', ofera doar metode pentru a initializa populatia.
    Functia 'initPopulation' are 1 parametru, population size.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self, config):
        super().__init__(config)

    def test(self, config):
        pass
        # test implementation
