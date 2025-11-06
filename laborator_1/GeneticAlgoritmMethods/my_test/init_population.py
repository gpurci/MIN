#!/usr/bin/python

import numpy as np

class InitPopulation(RootGA):
    """
    Clasa 'InitPopulation', ofera doar metode pentru a initializa populatia.
    Functia 'initPopulation' are 1 parametru, population size.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self, config):
        self.__config = config

    def __call__(self):
        fn = self.initPopulationAbstract
        if (self.__config == ""):
            fn = self.mutate
        return fn

    def initPopulationAbstract(self, parent1, parent2, offspring):
        raise NameError("Configuratie gresita pentru functia de 'InitPopulation'")
