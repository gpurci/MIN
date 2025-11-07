#!/usr/bin/python

import numpy as np
from my_code.root_GA import *
from math import hypot, ceil

class Metrics(RootGA):
    """
    Clasa 'Metrics', ofera doar metode pentru a calcula metrici pentru clase de probleme de optimizare.
    Functia 'metrics' are 1 parametru, populatia.
    Metoda '__config_fn', selecteaza functia ce calculeaza metricile.
    Metoda '__call__', aplica metrica ce a fost selectata in '__config_fn' asupra populatiei.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    # TO DO: sincronizare 'GENOME_LENGTH' cu celelalte clase!!!!!!
    def __init__(self, config):
        self.setConfig(config)

    def __call__(self, population):
        return self.fn(population)

    def __config_fn(self):
        self.fn = self.metricsAbstract
        if (self.__config is not None):
            if   (self.__config == "tsp"):
                self.fn = self.metricsTSP
        else:
            pass

    def setConfig(self, config):
        self.__config = config
        self.__config_fn()

    def setDataset(self, dataset):
        print("Utilizezi metoda: {}, datele de antrenare trebuie sa corespunda metodei de calcul a metricilor!!!".format(self.__config))
        # TO DO: seteaza 'GENOME_LENGTH'
        self.dataset = dataset

    def getDataset(self):
        return self.dataset

    def metricsAbstract(self, population):
        raise NameError("Lipseste configuratia pentru functia de 'Metrics': config '{}'".format(self.__config))

    # TS problem------------------------------
    def __getIndividDistance(self, individ):
        """Calculul distantei pentru un individ"""
        distances = self.dataset[individ[:-1], individ[1:]]
        distance  = distances.sum()
        return distance

    def __getIndividNumberCities(self, individ):
        return np.unique(individ[:-1], return_index=False, return_inverse=False, return_counts=False, axis=None).shape[0]

    def __getDistances(self, population):
        """Calculaza distanta pentru fiecare individ din populatiei"""
        return np.apply_along_axis(self.__getIndividDistance,
                                        axis=1,
                                        arr=population)

    def __getNumberCities(self, population):
        """Calculeaza numarul de orase pentru fiecare individ din populatie"""
        return np.apply_along_axis(self.__getIndividNumberCities,
                                        axis=1,
                                        arr=population)

    def metricsTSP(self, population):
        """ Returneaza o valoare normalizata, formula 2*weights*profits/(weights+profits)
        unde: valoarea distantei este invers normalizata, iar valoarea numarului de orase direct normalizata
        population - populatia, vector de indivizi
        """
        # calculeaza distanta
        distances   = self.__getDistances(population)
        # calculeaza numarul de orase unice
        number_city = self.__getNumberCities(population)
        metrics_values = {"distances": distances, "number_city":number_city}
        return metrics_values
    # TP problem finish =================================

    # TP problem metrics =========================
    def _pairwise_distance(self, coords: np.ndarray, ceil2d: bool = True) -> np.ndarray:
        n = coords.shape[0]
        D = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            xi, yi = coords[i]
            for j in range(i+1, n):
                xj, yj = coords[j]
                d = hypot(xi - xj, yi - yj)
                D[i, j] = D[j, i] = float(ceil(d)) if ceil2d else d
        return D
