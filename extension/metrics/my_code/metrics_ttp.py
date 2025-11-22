#!/usr/bin/python

import numpy as np
from extension.metrics.my_code.metrics_base import *

class MetricsTTP(MetricsBase):
    """
    Clasa 'Metrics', ofera doar metode pentru a calcula metrici pentru clase de probleme de optimizare.
    Functia 'metrics' are 1 parametru, populatia.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, dataset, **configs):
        super().__init__(method, name="MetricsTTP", **configs)
        self.__fn, self.getScore = self._unpackMethod(method, 
                                        TTP_linear=(self.metricsTTPLiniar, self.getScoreTTP), 
                                        TTP_ada_linear=(self.metricsTTPAdaLiniar, self.getScoreTTP),
                                        TTP_exp=(self.metricsTTPExp, self.getScoreTTP),
                                    )
        self.dataset = dataset

    def __call__(self, genomics):
        return self.__fn(genomics, **self._configs)

    def help(self):
        info = """MetricsTTP:
    metoda: 'TTP_linear';     config: -> "v_min":0.1, "v_max":1, "W":2000, "alpha":0.01;
    metoda: 'TTP_ada_linear'; config: -> "v_min":0.1, "v_max":1, "W":2000, "alpha":0.01;
    metoda: 'TTP_exp';        config: -> "v_min":0.1, "v_max":1, "W":2000, "lam":0.01;\n"""
        print(info)

    # individ metrics ---------------------
    def computeIndividProfitKP(self, kp_individ):
        return (self.dataset["item_profit"]*kp_individ).sum()

    def computeIndividWeightKP(self, kp_individ):
        return (self.dataset["item_weight"]*kp_individ).sum()

    def computeIndividNbrObjKP(self, kp_individ):
        return kp_individ.sum()

    def computeIndividNumberCities(self, individ):
        return np.unique(individ, return_index=False, return_inverse=False, return_counts=False, axis=None).shape[0]
        
    def computeIndividDistance(self, individ):
        """Calculul distantei pentru un individ"""
        #print("individ", individ.shape, end=", ")
        distance = self.dataset["distance"][individ[:-1], individ[1:]].sum()
        distance = distance + self.dataset["distance"][individ[-1], individ[0]]
        return distance
    # individ metrics =====================

    # population metrics ---------------------
    def computeDistances(self, population):
        """Calculaza distanta pentru fiecare individ din populatiei"""
        return np.apply_along_axis(self.computeIndividDistance,
                                        axis=1,
                                        arr=population)


    def computeNumberCities(self, population):
        """Calculeaza numarul de orase pentru fiecare individ din populatie"""
        return np.apply_along_axis(self.computeIndividNumberCities,
                                        axis=1,
                                        arr=population)

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
    # population metrics =====================


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
        number_city = self.computeNumberCities(genomics.chromosomes("tsp"))
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
        number_city = self.computeNumberCities(genomics.chromosomes("tsp"))
        #number_obj  = self.computeNbrObjKP(genomics.chromosomes("kp"))
        tmp_profits = self.computeProfitKP(genomics.chromosomes("kp"))

        W = kw.get("W")
        if (W < weights.min()):
            number_obj = np.mean(weights) / weights
            #number_obj = normalization(number_obj)+0.1
        else:
            number_obj = W / (weights + 1e-7)
            mask = number_obj > 1
            number_obj[mask] = 1/number_obj[mask]

        if (number_obj.max() < 10):
            number_obj = number_obj**5

        tmp_profits = tmp_profits / (tmp_profits.max() + 1e-7)
        number_obj *= tmp_profits
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

    def computeSpeedTTP(self, Wcur, v_max, v_min, Wmax):
        """
        Standard TTP speed: v = v_max - (v_max - v_min) * (Wcur / Wmax),
        clamped to [v_min, v_max].
        """
        v = v_max - (v_max - v_min) * (Wcur / float(Wmax))
        if v < v_min:
            v = v_min
        elif v > v_max:
            v = v_max
        return v
    
    def getIndividDistanceTTP(self, tsp_individ, distance=None):
        """
        Distance of a TSP tour 
        """
        if distance is None:
            distance = self.dataset["distance"]

        d = distance[tsp_individ[:-1], tsp_individ[1:]].sum()
        d += distance[tsp_individ[-1], tsp_individ[0]]
        return d


    def getScoreTTP(self, genomics, fitness_values):
        # find best individual
        arg_best = self.getArgBest(fitness_values)
        individ  = genomics[arg_best]
        best_fitness = fitness_values[arg_best]
        genomics.setBest(individ)
        distance = self.computeIndividDistance(individ["tsp"])
        kp_individ = individ["kp"]
        profit = self.computeIndividProfitKP(kp_individ)
        weight = self.computeIndividWeightKP(kp_individ)

        return {"score": profit, "distance":distance, "weight":weight, "best_fitness": best_fitness}
    # TTP problem finish =================================

def normalization(x):
    x_min = x.min()
    x_max = x.max()
    denom = x_max - x_min
    if (denom == 0):
        # all values are identical: return a constant vector (e.g. all 1s)
        x_ret = np.ones_like(x, dtype=np.float32)
    else:
        x_ret = (x_max-x)/(x_max-x_min)
    return x_ret
