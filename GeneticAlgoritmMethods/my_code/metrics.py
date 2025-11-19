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
        return self.__fn(genomics, **self.__configs)

    def help(self):
        info = """Metrics:
    metoda: 'TTP_linear'; config: -> "v_min":0.1, "v_max":1, "W":2000, "alpha":0.01;
    metoda: 'TTP_ada_linear'; config: -> "v_min":0.1, "v_max":1, "W":2000, "alpha":0.01;
    metoda: 'TTP_exp';    config: -> "v_min":0.1, "v_max":1, "W":2000, "lam":0.01;
    metoda: 'TSP';        config: None;
    metoda: 'extern';     config: 'extern_kw';\n"""
        return info

    def __unpackMethod(self, method, extern_fn):
        fn = self.metricsAbstract
        if (method is not None):
            if   (method == "TSP"):
                fn = self.metricsTSP
                self.getScore = self.getScoreTSP
            elif (method == "TTP_linear"):
                fn = self.metricsTTPLiniar
                self.getScore = self.getScoreTTP
            elif (method == "TTP_ada_linear"):
                fn = self.metricsTTPAdaLiniar
                self.getScore = self.getScoreTTP
            elif (method == "TTP_exp"):
                fn = self.metricsTTPExp
                self.getScore = self.getScoreTTP
            elif ((method == "extern") and (extern_fn is not None)):
                fn = extern_fn
                self.getScore = extern_fn.getScore

        return fn

    def __setMethods(self, method):
        self.__method = method
        extern_fn = self.__configs.pop("extern_fn", None)
        self.__fn = self.__unpackMethod(method, extern_fn)

    def setDataset(self, dataset):
        print("Utilizezi metoda: {}, datele de antrenare trebuie sa corespunda metodei de calcul a metricilor!!!".format(self.__method))
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
        raise NameError("Lipseste metoda '{}' pentru functia de 'Metrics': config '{}'".format(self.__method, self.__configs))

    # TS problem------------------------------
    def __getIndividDistance(self, individ):
        """Calculul distantei pentru un individ"""
        #print("individ", individ.shape, end=", ")
        distances = self.dataset["distance"][individ[:-1], individ[1:]]
        distance  = distances.sum() + self.dataset["distance"][individ[-1], individ[0]]
        return distance

    def __getIndividNumberCities(self, individ):
        return np.unique(individ, return_index=False, return_inverse=False, return_counts=False, axis=None).shape[0]

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
    def computeIndividProfitKP(self, kp_individ):
        return (self.dataset["item_profit"]*kp_individ).sum()

    def computeIndividWeightKP(self, kp_individ):
        return (self.dataset["item_weight"]*kp_individ).sum()

    def computeIndividNbrObjKP(self, kp_individ):
        return kp_individ.sum()

    def computeProfitKP(self, kp_population):
        """
        Calculeaza profitul pentru intrega populatie
        """
        return np.apply_along_axis(self.computeIndividProfitKP,
                                        axis=1,
                                        arr=kp_population)

    def computeWeightKP(self, kp_population):
        """
        Calculeaza profitul pentru intrega populatie
        """
        return np.apply_along_axis(self.computeIndividWeightKP,
                                        axis=1,
                                        arr=kp_population)

    def computeNbrObjKP(self, kp_population):
        """
        Calculeaza profitul pentru intrega populatie
        """
        return np.apply_along_axis(self.computeIndividNbrObjKP,
                                        axis=1,
                                        arr=kp_population)
    #  TTP Liniar ---------------------
    def __computeIndividLiniarTTP(self, individ, *args, v_min=0.1, v_max=1, W=2000, alpha=0.01):
        # init 
        Wcur = 0.0
        Tcur = 0.0
        Pcur = 0.0
        # unpack chromosomes
        tsp_individ = individ["tsp"]
        kp_individ  = individ["kp"]
        # unpack args
        # unpack datassets
        distance, item_profit, item_weight = args
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
            v = v_max - v_min * (Wcur / W)
            v = max(v_min, v)
            # calculeaza timpul de tranzitie
            Tcur += distance[city, tsp_individ[i+1]] / v
        # intorcerea in orasul de start
        # calculeaza viteza
        Tcur += distance[tsp_individ[-1], tsp_individ[0]] / v
        return Pcur, Tcur, Wcur

    def metricsTTPLiniar(self, genomics, **kw):
        # unpack datassets
        distance    = self.dataset["distance"]
        item_profit = self.dataset["item_profit"]
        item_weight = self.dataset["item_weight"]
        # pack args
        args = [distance, item_profit, item_weight]
        # calculate metrics for every individ
        profits = np.zeros(self.POPULATION_SIZE, dtype=np.float32)
        weights = np.zeros(self.POPULATION_SIZE, dtype=np.float32)
        times   = np.zeros(self.POPULATION_SIZE, dtype=np.float32)
        for idx, individ in enumerate(genomics.population(), 0):
            profit, time, weight = self.__computeIndividLiniarTTP(individ, *args, **kw)
            profits[idx] = profit
            weights[idx] = weight
            times[idx]   = time

        # number city
        number_city = self.__getNumberCities(genomics.chromozomes("tsp"))
        # pack metrick values
        metric_values = {
            "profits"    : profits,
            "times"      : times,
            "weights"    : weights,
            "number_city": number_city
        }
        return metric_values
    # TTP Liniar =================================

    #  TTP Liniar ---------------------
    def __computeIndividAdaLiniarTTP(self, individ, *args, v_min=0.1, v_max=1, W=2000, alpha=0.01):
        # init 
        Wcur = 0.0
        Tcur = 0.0
        Pcur = 0.0
        # unpack chromosomes
        tsp_individ = individ["tsp"]
        kp_individ  = individ["kp"]
        # unpack args
        # unpack datassets
        distance, item_profit, item_weight = args
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
            Pcur += max(0.0, profit**2/(weight+1e-7) - alpha*Tcur)
            Wcur += weight
            # calculeaza viteza de tranzitie
            v = v_max - v_min * ((Wcur / W) - 1.)
            v = max(v_min, v)
            # calculeaza timpul de tranzitie
            Tcur += distance[city, tsp_individ[i+1]] / v
        # intorcerea in orasul de start
        # calculeaza viteza
        Tcur += distance[tsp_individ[-1], tsp_individ[0]] / v
        return Pcur, Tcur, Wcur

    def metricsTTPAdaLiniar(self, genomics, **kw):
        # unpack datassets
        distance    = self.dataset["distance"]
        item_profit = self.dataset["item_profit"]
        item_weight = self.dataset["item_weight"]
        # pack args
        args = [distance, item_profit, item_weight]
        # calculate metrics for every individ
        profits = np.zeros(self.POPULATION_SIZE, dtype=np.float32)
        weights = np.zeros(self.POPULATION_SIZE, dtype=np.float32)
        times   = np.zeros(self.POPULATION_SIZE, dtype=np.float32)
        for idx, individ in enumerate(genomics.population(), 0):
            profit, time, weight = self.__computeIndividAdaLiniarTTP(individ, *args, **kw)
            profits[idx] = profit
            weights[idx] = weight
            times[idx]   = time

        # number city
        number_city = self.__getNumberCities(genomics.chromozomes("tsp"))
        number_obj  = self.computeNbrObjKP(genomics.chromozomes("kp"))
        number_obj  = number_obj*kw.get("W")/(weights*self.GENOME_LENGTH + 1e-7)

        mask = (kw.get("W") <= weights)
        number_obj[mask] = 1.

        # pack metrick values
        metric_values = {
            "profits"    : profits,
            "times"      : times,
            "weights"    : weights,
            "number_city": number_city,
            "number_obj" : number_obj
        }
        return metric_values
    # TTP Liniar =================================
    
    #  TTP Exponential ---------------------
    def __computeIndividExpTTP(self, individ, *args, v_min=0.1, v_max=1, W=2000, lam=0.01):
        # init 
        Wcur = 0.0
        Tcur = 0.0
        Pcur = 0.0
        # unpack chromosomes
        tsp_individ = individ["tsp"]
        kp_individ  = individ["kp"]
        # unpack args
        # unpack datassets
        distance, item_profit, item_weight = args
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
            Pcur += profit * np.exp(-lam * Tcur)
            Wcur += weight
            # calculeaza viteza de tranzitie
            v = v_max - (v_max - v_min) * (Wcur / W)
            # calculeaza timpul de tranzitie
            Tcur += distance[city, tsp_individ[i+1]] / v
        # intorcerea in orasul de start
        # calculeaza viteza
        v = v_max - (v_max - v_min) * (Wcur / W)
        Tcur += distance[tsp_individ[-1], tsp_individ[0]] / v
        return Pcur, Tcur, Wcur

    def metricsTTPExp(self, genomics, **kw):
        # unpack datassets
        distance    = self.dataset["distance"]
        item_profit = self.dataset["item_profit"]
        item_weight = self.dataset["item_weight"]
        # pack args
        args = [distance, item_profit, item_weight]
        # calculate metrics for every individ
        profits = np.zeros(self.POPULATION_SIZE, dtype=np.float32)
        weights = np.zeros(self.POPULATION_SIZE, dtype=np.float32)
        times   = np.zeros(self.POPULATION_SIZE, dtype=np.float32)
        for idx, individ in enumerate(genomics.population(), 0):
            profit, time, weight = self.__computeIndividExpTTP(individ, *args, **kw)
            profits[idx] = profit
            weights[idx] = weight
            times[idx]   = time
        # pack metrick values
        metric_values = {
            "profits"  : profits,
            "weights"  : weights,
            "times"    : times
        }
        return metric_values
    # TTP Exponential =================================

    def getScoreTTP(self, genomics, fitness_values):
        # find best individual
        arg_best = self.getArgBest(fitness_values)
        individ  = genomics[arg_best]
        best_fitness = fitness_values[arg_best]
        self.__best_individ = individ
        distance = self.__getIndividDistance(individ["tsp"])
        kp_individ = individ["kp"]
        profit = self.computeIndividProfitKP(kp_individ)
        weight = self.computeIndividWeightKP(kp_individ)

        return {"score": profit, "distance":distance, "weight":weight, "best_fitness": best_fitness}
    # TTP problem finish =================================
