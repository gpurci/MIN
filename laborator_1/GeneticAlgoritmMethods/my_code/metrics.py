#!/usr/bin/python

import numpy as np
from root_GA import *

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
        super().__init__()
        
        self.setConfig(config)

    def __call__(self, population):
        return self.fn(population)

    def help(self):
        info = """Metrics: 
        metode de config: 'TSP'\n"""
        return info

    def __config_fn(self):
        self.fn = self.metricsAbstract
        if (self.__config is not None):
            if   (self.__config == "TSP"):
                self.fn = self.metricsTSP
                self.getScore = self.getScoreTSP
        else:
            pass

    def setConfig(self, config):
        self.__config = config
        self.__config_fn()

    def setDataset(self, dataset):
        print("Utilizezi metoda: {}, datele de antrenare trebuie sa corespunda metodei de calcul a metricilor!!!".format(self.__config))
        self.dataset = dataset

    def getDataset(self):
        return self.dataset

    def getArgBest(self, fitness_values):
        """Cautarea rutei optime din populatie"""
        index = np.argmax(fitness_values, axis=None, keepdims=False)
        return index

    def getBestIndivid(self):
        return self.__best_individ

    def metricsAbstract(self, population):
        raise NameError("Lipseste configuratia pentru functia de 'Metrics': config '{}'".format(self.__config))

    # TS problem------------------------------
    def __getIndividDistance(self, individ):
        """Calculul distantei pentru un individ"""
        distances = self.dataset[individ[:-1], individ[1:]]
        distance  = distances.sum() + self.dataset[individ[-1], individ[0]]
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
        metric_values = {"distances": distances, "number_city":number_city}
        return metric_values

    def getScoreTSP(self, population, fitness_values):
        # obtinerea celui mai bun individ
        arg_best = self.getArgBest(fitness_values)
        individ  = population[arg_best]
        best_fitness = fitness_values[arg_best]
        self.__best_individ = individ
        score = self.__getIndividDistance(individ)
        return {"score": score, "best_fitness": best_fitness}

    # TSP problem finish =================================

    # TTP problem metrics ---------------------
    # calculeaza distante perechi, aplicand CEIL_2D (rotunjire în sus, nu distanta euclidiana reala)
    def _pairwise_distance(self, coords, is_ceil2d: bool = True) -> np.ndarray:
        map_of_distance = []
        for point in coords:
            tmp_distance = np.linalg.norm(coords - point, axis=1)
            map_of_distance.append(tmp_distance)
        map_of_distance = np.array(map_of_distance)
        if is_ceil2d:
            map_of_distance = np.round(map_of_distance, 0)
        return map_of_distance

    
    def computeSpeedTTP(self, Wcur, vmax, vmin, Wmax):
        """
        viteza curenta in functie de weight (formula TTP)
        """
        frac = min(1.0, Wcur/Wmax)
        v = vmax - frac*(vmax-vmin)
        return v if v > 1e-9 else 1e-9

    def getIndividDistanceTTP(self, individ, distance_matrix=None):
        """
        distanta rutelor TTP (daca inchizi ruta)
        use: metrics.getIndividDistanceTTP(individ)
        """
        D = distance_matrix if distance_matrix is not None else self.dataset
        distances = D[individ[:-1], individ[1:]]
        return distances.sum() + D[individ[-1], individ[0]]
    # TTP problem finish =================================
