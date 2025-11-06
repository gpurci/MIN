#!/usr/bin/python

import numpy as np
from my_code.root_GA import *

class InitPopulation(RootGA):
    """
    Clasa 'InitPopulation', ofera doar metode pentru a initializa populatia.
    Functia 'initPopulation' are 1 parametru, population size.
    Metoda '__config_fn', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self, config):
        self.__config = config.get("init_population", None)
        self.__config_fn()

    def __config_fn(self):
        self.initPopulation = self.initPopulationAbstract
        if (self.__config is not None):
            if   (self.__config == "test"):
                self.initPopulation = self.testParentClass
        else:
            pass

    def initPopulationAbstract(self, size):
        raise NameError("Configuratie gresita pentru functia de 'InitPopulation'")
