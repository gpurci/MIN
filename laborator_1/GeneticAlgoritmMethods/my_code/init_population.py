#!/usr/bin/python

import numpy as np
from my_code.root_GA import *

class InitPopulation(RootGA):
    """
    Clasa 'InitPopulation', ofera doar metode pentru a initializa populatia.
    Functia 'initPopulation' are 1 parametru, numarul populatiei.
    Metoda '__config_fn', selecteaza functia de initializare.
    Metoda '__call__', aplica functia de initializare ce a fost selectata in '__config_fn'
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self, config, metrics):
        self.metrics = metrics
        self.setConfig(config)

    def __call__(self, size):
        return self.fn(size)

    def __config_fn(self):
        self.fn = self.initPopulationAbstract
        if (self.__config is not None):
            if   (self.__config == "test"):
                self.fn = self.testParentClass
        else:
            pas

    def setConfig(self, config):
        self.__config = config
        self.__config_fn()

    def initPopulationAbstract(self, size):
        raise NameError("Lipseste configuratia pentru functia de 'InitPopulation': config '{}'".format(self.__config))
