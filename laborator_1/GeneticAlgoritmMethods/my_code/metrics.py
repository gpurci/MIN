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
    def __init__(self, method, **kw):
        super().__init__()
        self.__configs = kw
        self.__setMethods(method)

    def __str__(self):
        info = """Metrics: 
        method:  {}
        configs: {}""".format(self.__method, self.__configs)
        return info

    def __call__(self, genomics):
        return self.fn(genomics, **self.__configs)

    def help(self):
        info = """Metrics: 
    metoda: 'TTP'; config: -> lambda_time, vmax, vmin, Wmax, seed;
    metoda: 'TSP'; config: None;\n"""
        return info

    def __method_fn(self):
        self.fn = self.metricsAbstract
        if (self.__method is not None):
            if   (self.__method == "TSP"):
                self.fn = self.metricsTSP
                self.getScore = self.getScoreTSP
            elif (self.__method == "TTP"):
                self.fn = self.metricsTTP
                self.getScore = self.getScoreTTP


    def __setMethods(self, method):
        self.__method = method
        self.__method_fn()

    def setDataset(self, dataset):
        print("Utilizezi metoda: {}, datele de antrenare trebuie sa corespunda metodei de calcul a metricilor!!!".format(self.__method))
        self.dataset = dataset

        if   (self.__method == "TSP"):
            # dataset este direct matrice NxN
            self.distance = dataset

        elif (self.__method == "TTP"):
            # dataset este dict
            self.coords      = dataset["coords"]
            self.distance    = dataset["distance"]
            self.item_profit = dataset["item_profit"]
            self.item_weight = dataset["item_weight"]

            # pentru fitness (city, w, p)
            self.items = list(zip(
                np.arange(len(self.item_profit)),
                self.item_weight,
                self.item_profit
            ))


    def getDataset(self):
        return self.dataset

    def getArgBest(self, fitness_values):
        """Cautarea rutei optime din populatie"""
        index = np.argmax(fitness_values, axis=None, keepdims=False)
        return index

    def getBestIndivid(self):
        return self.__best_individ

    def metricsAbstract(self, population):
        raise NameError("Lipseste metoda '{}' pentru functia de 'Metrics': config '{}'".format(self.__method, self.__configs))

    # TS problem------------------------------
    def __getIndividDistance(self, individ):
        """Calculul distantei pentru un individ"""
        #print("individ", individ.shape, end=", ")
        distances = self.dataset["distance"][individ[:-1], individ[1:]]
        distance  = distances.sum() + self.dataset["distance"][individ[-1], individ[0]]
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

    def metricsTSP(self, genomics):
        """ Returneaza o valoare normalizata, formula 2*weights*profits/(weights+profits)
        unde: valoarea distantei este invers normalizata, iar valoarea numarului de orase direct normalizata
        population - populatia, vector de indivizi
        """
        population = genomics.population("tsp")
        #print("\nmetricsTSP, genomics: {}, population {}, last {}".format(genomics.shape, population.shape, population[-1].shape))
        #print("\nmetricsTSP, population {}".format(population))
        # calculeaza distanta
        distances   = self.__getDistances(population)
        # calculeaza numarul de orase unice
        number_city = self.__getNumberCities(population)
        metric_values = {"distances": distances, "number_city":number_city}
        return metric_values

    def getScoreTSP(self, genomics, fitness_values):
        # obtinerea celui mai bun individ
        arg_best = self.getArgBest(fitness_values)
        individ  = genomics[arg_best]["tsp"]
        best_fitness = fitness_values[arg_best]
        self.__best_individ = individ
        score = self.__getIndividDistance(individ)
        return {"score": score, "best_fitness": best_fitness}

    # TSP problem finish =================================

    # TTP problem metrics ---------------------
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
        D = distance_matrix if (distance_matrix is not None) else self.distance

        distances = D[individ[:-1], individ[1:]]
        return distances.sum() + D[individ[-1], individ[0]]
    
    def metricsTTP(self, population):
        N = population.shape[0]
        distances = np.zeros(N, dtype=np.float32)
        profits   = np.zeros(N, dtype=np.float32)
        times     = np.zeros(N, dtype=np.float32)

        for i, ind in enumerate(population):

            Wcur  = 0.0
            T     = 0.0
            P     = 0.0

            for k in range(len(ind)-1):
                city = ind[k]

                # adds profit
                for (city_k, weight_k, profit_k) in self.items:
                    if city_k == city:
                        P += profit_k
                        Wcur += weight_k

                v = self.computeSpeedTTP(Wcur, self.v_max, self.v_min, self.W)
                T += self.distance[ind[k], ind[k+1]] / v

            distances[i] = self.getIndividDistanceTTP(ind)
            profits[i]   = P
            times[i]     = T

        self.metrics_values = {
            "distances" : distances,
            "profits"   : profits,
            "times"     : times
        }
        return self.metrics_values
    
    def getScoreTTP(self, population, fitness_values):
        # find best individual
        arg_best = self.getArgBest(fitness_values)
        individ  = population[arg_best]
        best_fitness = fitness_values[arg_best]
        self.__best_individ = individ

        score = self.getIndividDistanceTTP(individ, self.distance)

        return {"score": score, "best_fitness": best_fitness}
    # TTP problem finish =================================
