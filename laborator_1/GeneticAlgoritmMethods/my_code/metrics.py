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
        population = genomics.chromozomes("tsp")
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
        v    = vmax - frac*(vmax-vmin)
        return v if v > 1e-9 else 1e-9

    def computeIndividProfitKP(self, kp_individ):
        return (self.dataset["item_profit"]*kp_individ).sum()

    def computeIndividWeightKP(self, kp_individ):
        return (self.dataset["item_weight"]*kp_individ).sum()

    def computeProfitTTP(self, kp_population):
        """
        Calculeaza profitul pentru intrega populatie
        """
        return np.apply_along_axis(self.computeIndividProfitKP,
                                        axis=1,
                                        arr=kp_population)

    def __computeIndividLiniarTTP(self, individ, v_min, v_max, W, alpha):
        # init 
        Wcur = 0.0
        Tcur = 0.0
        Pcur = 0.0
        # unpack chromosomes
        tsp_individ = individ["tsp"]
        kp_individ  = individ["kp"]
        # unpack datassets
        distance    = self.dataset["distance"]
        item_profit = self.dataset["item_profit"]
        item_weight = self.dataset["item_weight"]
        # vizităm secvenţial
        for i in range(self.GENOME_LENGTH-1):
            # get city
            city = tsp_individ[i]
            # take or not take object
            take = kp_individ[city]
            # calculate profit and weight
            profit = item_profit[city]*take
            weight = item_weight[city]*take
            # calculate liniar profit and weight
            Pcur += max(0.0, profit - alpha*Tcur)
            Wcur += weight
            # calculeaza viteza de tranzitie
            v = v_max - (v_max - v_min) * (Wcur / W)
            # calculeaza timpul de tranzitie
            Tcur += distance[city, tsp_individ[i+1]] / v
        # intorcerea in orasul de start
        # calculeaza viteza
        v = v_max - (v_max - v_min) * (Wcur / W)
        Tcur += distance[individ[-1], individ[0]] / v
        return Pcur, Tcur


    def metricsTTPLiniar(self, genomics):
        # calculate distances
        distances = self.__getDistances(tsp_population)
        profits, times = np.apply_along_axis(self.__computeIndividLiniarTTP,
                                        axis=1,
                                        arr=genomics.population())

        metric_values = {
            "distances": distances,
            "profits"  : profits,
            "times"    : times
        }
        return metric_values
    
    def getScoreTTP(self, genomics, fitness_values):
        # find best individual
        arg_best = self.getArgBest(fitness_values)
        individ  = genomics[arg_best]
        best_fitness = fitness_values[arg_best]
        self.__best_individ = individ
        score  = self.__getIndividDistance(individ["tsp"])
        kp_individ = individ["kp"]
        profit = self.computeIndividProfitKP(kp_individ)
        weight = self.computeIndividWeightKP(kp_individ)

        return {"score": score, "profit":profit, "weight":weight, "best_fitness": best_fitness}
    # TTP problem finish =================================
