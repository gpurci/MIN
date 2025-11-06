#!/usr/bin/python

import numpy as np
from my_code.root_GA import *

class Fitness(RootGA):
    """
    Clasa 'Fitness', ofera doar metode pentru a calcula functia fitness din populatie.
    Functia 'fitness' are 1 parametru, numarul populatiei.
    Metoda '__config_fn', selecteaza functia de initializare.
    Metoda '__call__', aplica functia de initializare ce a fost selectata in '__config_fn'
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self, config):
        self.setConfig(config)

    def __call__(self, size):
        return self.fn(size)

    def __config_fn(self):
        self.fn = self.fitnessAbstract
        if (self.__config is not None):
            if   (self.__config == "f1score"):
                self.fn = self.fitnessF1score
        else:
            pas

    def setConfig(self, config):
        self.__config = config
        self.__config_fn()

    def setTrainData(self, train_ds, ):
        self.train_ds = train_ds

    def fitnessAbstract(self, size):
        raise NameError("Configuratie gresita pentru functia de 'Fitness'")

    # TS problem------------------------------
    def getIndividDistance(self, individ):
        """Calculul distantei pentru un individ"""
        distances = self.train_ds[individ[:-1], individ[1:]]
        distance  = distances.sum()
        return distance

    def getIndividNumberCities(self, individ):
        return np.unique(individ[:-1], return_index=False, return_inverse=False, return_counts=False, axis=None).shape[0]

    def getBestNumberCities(self, population):
        """Calculeaza cel mai mare numar de orase din intreaga populatie"""
        number_city = self.getNumberCities(population)
        return number_city.max()

    def getDistances(self, population):
        """Calculaza distanta pentru fiecare individ din populatiei"""
        return np.apply_along_axis(self.getIndividDistance,
                                        axis=1,
                                        arr=population)

    def getNumberCities(self, population):
        """Calculeaza numarul de orase pentru fiecare individ din populatie"""
        return np.apply_along_axis(self.getIndividNumberCities,
                                        axis=1,
                                        arr=population)

    def fitnessF1score(self, population):
        """ Returneaza o valoare normalizata, formula 2*weights*profits/(weights+profits)
        unde: valoarea distantei este invers normalizata, iar valoarea numarului de orase direct normalizata
        population - populatia, vector de indivizi
        """
        # calculeaza distanta
        distances   = self.getDistances(population)
        # calculeaza numarul de orase unice
        number_city = self.getNumberCities(population)
        # normalizeaza intervalul 0...1
        number_city = self.cityNorm(number_city)
        distances   = self.distanceNorm(distances)
        fitness_values = 2*distances*number_city/(distances+number_city+1e-7)
        return fitness_values

    def distanceNorm(self, distances):
        mask_not_zero   = (distances!=0)
        valid_distances = distances[mask_not_zero]
        if (valid_distances.shape[0] > 0):
            self.__min_distance = valid_distances.min()
        else:
            self.__min_distance = 0.1
            distances[:] = 0.1
        return (2*self.__min_distance)/(distances+self.__min_distance)

    def cityNorm(self, number_city):
        mask_cities = (number_city==Fitness.GENOME_LENGTH)
        return mask_cities.astype(np.float32)
    # TP problem finish
