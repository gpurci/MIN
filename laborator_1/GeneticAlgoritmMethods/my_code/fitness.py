#!/usr/bin/python

import numpy as np
from root_GA import *

class Fitness(RootGA):
    """
    Clasa 'Fitness', ofera doar metode pentru a calcula functia fitness din populatie.
    Functia 'fitness' are 1 parametru, numarul populatiei.
    Metoda '__config_fn', selecteaza functia de initializare.
    Metoda '__call__', aplica functia de initializare ce a fost selectata in '__config_fn'
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self, config):
        super().__init__()
        self.setConfig(config)

    def __call__(self, population, metric_values):
        return self.fn(population, metric_values)

    def __config_fn(self):
        self.fn = self.fitnessAbstract
        if (self.__config is not None):
            if   (self.__config == "TSP_f1score"):
                self.fn = self.fitnessF1scoreTSP
        else:
            pass

    def help(self):
        info = """Fitness: 
        metode de config: 'TSP_f1score'\n"""
        return info

    def setConfig(self, config):
        self.__config = config
        self.__config_fn()

    def fitnessAbstract(self, population):
        raise NameError("Lipseste configuratia pentru functia de 'Fitness': config '{}'".format(self.__config))

    # TSP problem------------------------------
    def fitnessF1scoreTSP(self, population, metric_values):
        """ Returneaza o valoare normalizata, formula 2*weights*profits/(weights+profits)
        unde: valoarea distantei este invers normalizata, iar valoarea numarului de orase direct normalizata
        population - populatia, vector de indivizi
        """
        # despacheteaza metricile
        distances   = metric_values["distances"]
        number_city = metric_values["number_city"]
        # normalizeaza intervalul 0...1
        #print("number_city {}".format(number_city))
        number_city = self.__cityNormTSP(number_city)
        #print("number_city {}".format(number_city.sum()))
        distances   = self.__distanceNormTSP(distances)
        fitness_values = 2*distances*number_city/(distances+number_city+1e-7)
        #print("fitness {}".format(fitness_values))
        return fitness_values

    def __distanceNormTSP(self, distances):
        mask_not_zero   = (distances!=0)
        valid_distances = distances[mask_not_zero]
        if (valid_distances.shape[0] > 0):
            self.__min_distance = valid_distances.min()
        else:
            self.__min_distance = 0.1
            distances[:] = 0.1
        return (2*self.__min_distance)/(distances+self.__min_distance)

    def __cityNormTSP(self, number_city):
        mask_cities = (number_city>=(self.GENOME_LENGTH-5)).astype(np.float32)

        return mask_cities
    # TSP problem=================================
