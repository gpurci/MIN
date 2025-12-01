#!/usr/bin/python

import numpy as np
from extension.metrics.my_code.metrics_base import *
from extension.utils.my_code.normalization import *

class MetricsTTP(MetricsBase):
    """
    Clasa 'Metrics', ofera doar metode pentru a calcula metrici pentru clase de probleme de optimizare.
    Functia 'metrics' are 1 parametru, populatia.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, dataset_man, **configs):
        super().__init__(method, name="MetricsTTP", **configs)
        self.__fn, self.getScore = self._unpackMethod(method, 
                                        TTP_linear=(self.metricsTTPLiniar, self.getScoreTTPLiniar), 
                                        TTP_ada_linear=(self.metricsTTPAdaLiniar, self.getScoreTTPAdaLiniar),
                                        TTP_exp=(self.metricsTTPExp, None),
                                    )
        self.dataset_man = dataset_man

    def __call__(self, genomics):
        return self.__fn(genomics, **self._configs)

    def help(self):
        info = """MetricsTTP:
    metoda: 'TTP_linear';     config: -> "v_min":0.1, "v_max":1, "W":2000, "alpha":0.01;
    metoda: 'TTP_ada_linear'; config: -> "v_min":0.1, "v_max":1, "W":2000, "alpha":0.01;
    metoda: 'TTP_exp';        config: -> "v_min":0.1, "v_max":1, "W":2000, "lam":0.01;\n"""
        print(info)

    #  TTP Liniar ---------------------
    def __computeIndividLiniarTTP(self, individ, *args, v_min=0.1, v_max=1, W=2000, alpha=0.01, R=1):
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
            v = self.computeSpeedTTP(v_max, v_min, Wcur, W)
            # calculeaza timpul de tranzitie
            Tcur += distance[city, tsp_individ[i+1]] / v
        # intorcerea in orasul de start
        # calculeaza viteza
        Tcur += distance[tsp_individ[-1], tsp_individ[0]] / v
        return Pcur, Tcur, Wcur

    def metricsTTPLiniar(self, genomics, **kw):
        # unpack datassets
        distance, item_profit, item_weight = self.dataset_man.getTupleDataset()
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
        number_city = self.dataset_man.computeNumberCities(genomics.chromosomes("tsp"))

        W = kw.get("W")
        if (W < weights.min()):
            CAPACITY = np.mean(weights)
            weights  = CAPACITY / weights
        else:
            CAPACITY = W
            weights  = CAPACITY / (weights + 1e-7)
            mask     = weights > 1
            weights[mask] = 1./weights[mask]

        # pack metrick values
        metric_values = {
            "profits"    : profits,
            "times"      : times,
            "weights"    : weights,
            "number_city": number_city,
            "number_obj" : 1
        }
        return metric_values
    # TTP Liniar =================================

    #  TTP Liniar ---------------------
    def __computeIndividAdaLiniarTTP(self, individ, *args, v_min=0.1, v_max=1, W=2000, alpha=0.01, R=1):
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
        distance, item_profit, item_weight = self.dataset_man.getTupleDataset()
        # pack args
        args = [distance, item_profit, item_weight]
        # calculate metrics for every individ
        profits = np.zeros(genomics.shape[0], dtype=np.float32)
        weights = np.zeros(genomics.shape[0], dtype=np.float32)
        times   = np.zeros(genomics.shape[0], dtype=np.float32)
        for idx, individ in enumerate(genomics.population(), 0):
            profit, time, weight = self.__computeIndividAdaLiniarTTP(individ, *args, **kw)
            profits[idx] = profit
            weights[idx] = weight
            times[idx]   = time

        # number city
        number_city = self.dataset_man.computeNumberCities(genomics.chromosomes("tsp"))
        number_obj  = self.dataset_man.computeNbrObj(genomics.chromosomes("kp"))

        W = kw.get("W")
        if (W < weights.min()):
            CAPACITY = np.mean(weights)
            weights  = CAPACITY / weights
        else:
            CAPACITY = W
            weights  = CAPACITY / (weights + 1e-7)
            mask     = weights > 1
            weights[mask] = 1./weights[mask]

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
    
    # TTP problem finish =================================
    
    # Calculate score ++++++++++++++++++++++++++++++
    def getScoreTTPLiniar(self, genomics, fitness_values):
        # find best individual
        arg_best = self.getArgBest(fitness_values)
        # get individ
        individ  = genomics[arg_best]
        # get best fitness
        best_fitness = fitness_values[arg_best]
        # set best individ
        genomics.setBest(individ)
        # conpute distance
        distance = self.dataset_man.computeIndividDistance(individ["tsp"])
        # compute profit and weight
        kp_individ = individ["kp"]
        profit = self.dataset_man.computeIndividProfit(kp_individ)
        weight = self.dataset_man.computeIndividWeight(kp_individ)
        # unpack datassets
        args   = [*self.dataset_man.getTupleDataset()]
        p, t, w  = self.__computeIndividLiniarTTP(individ, *args, **self._configs)
        score  = p - self._configs.get("R")*t
        return {"score": score, 
        "profit":profit, 
        "profit_ttp":p, 
        "distance":distance, 
        "time_ttp":t, 
        "weight":weight, 
        "weight_ttp":w, 
        "best_fitness":best_fitness}

    def getScoreTTPAdaLiniar(self, genomics, fitness_values):
        # find best individual
        arg_best = self.getArgBest(fitness_values)
        # get individ
        individ  = genomics[arg_best]
        # get best fitness
        best_fitness = fitness_values[arg_best]
        # set best individ
        genomics.setBest(individ)
        # conpute distance
        distance = self.dataset_man.computeIndividDistance(individ["tsp"])
        # compute profit and weight
        kp_individ = individ["kp"]
        profit = self.dataset_man.computeIndividProfit(kp_individ)
        weight = self.dataset_man.computeIndividWeight(kp_individ)
        # unpack datassets
        args   = [*self.dataset_man.getTupleDataset()]
        p, t, w  = self.__computeIndividAdaLiniarTTP(individ, *args, **self._configs)
        score  = p - self._configs.get("R")*t
        return {"score": score, 
        "profit":profit, 
        "profit_ttp":p, 
        "distance":distance, 
        "time_ttp":t, 
        "weight":weight, 
        "weight_ttp":w, 
        "best_fitness":best_fitness}
    # Calculate score ==============================

    # utils ++++++++++++++++++++++++++++++++++++++++++++++
    def computeSpeedTTP(self, v_max, v_min, Wcur, Wmax):
        """
        Standard TTP speed: v = v_max - (v_max - v_min) * (Wcur / Wmax),
        clamped to [v_min, v_max].
        """
        v = v_max - (v_max - v_min) * (Wcur / Wmax)
        if v < v_min:
            v = v_min
        elif v > v_max:
            v = v_max
        return v
    # utils ==============================================
