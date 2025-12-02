#!/usr/bin/python

import numpy as np
from extension.metrics.my_code.metrics_base import *
from extension.utils.my_code.normalization import *

class MetricsTTPV1(MetricsBase):
    """
    Clasa 'Metrics', ofera doar metode pentru a calcula metrici pentru clase de probleme de optimizare.
    Functia 'metrics' are 1 parametru, populatia.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, dataset_man, **configs):
        super().__init__(method, name="MetricsTTPV1", **configs)
        self.__fn, self.getScore = self._unpackMethod(method, 
                                        liniar=(self.liniar, self.getScoreLiniar), 
                                    )
        self.dataset_man = dataset_man

    def __call__(self, genomics):
        return self.__fn(genomics, **self._configs)

    def help(self):
        info = """MetricsTTPV1:
    metoda: 'liniar'; config: -> v_min=0.1, v_max=1, W=2000, R=1;
    dataset_man - managerul setului de date\n"""
        print(info)

    # Liniar ---------------------
    def __computeIndividSpeeds(self, individ, *args, v_min=0.1, v_max=1, W=2000, R=1):
        # unpack chromosomes
        tsp_individ = individ["tsp"]
        kp_individ  = individ["kp"]
        # unpack args; datasets
        distance, item_weight = args
        # init viteza
        speeds = np.zeros_like(tsp_individ, dtype=np.float32)
        Wcur   = 0
        # vizităm secvenţial
        for i in range(self.GENOME_LENGTH):
            # get city
            city = tsp_individ[i]
            # take or not take object
            take = kp_individ[city]
            # calculate weight
            weight = item_weight[city]*take
            Wcur  += weight
            # calculeaza viteza de tranzitie
            speed     = v_max - (v_max - v_min) * (Wcur / W)
            speeds[i] = max(v_min, speed)
        return speeds

    def liniar(self, genomics, **kw):
        # unpack datassets
        distance, item_profit, item_weight = self.dataset_man.getTupleDataset()
        # pack args
        args = [distance, item_weight]
        # calculate metrics for every individ
        profits   = np.zeros(self.POPULATION_SIZE, dtype=np.float32)
        weights   = np.zeros(self.POPULATION_SIZE, dtype=np.float32)
        times     = np.zeros(self.POPULATION_SIZE, dtype=np.float32)
        distances = np.zeros(self.POPULATION_SIZE, dtype=np.float32)
        for idx, individ in enumerate(genomics.population(), 0):
            speeds = self.__computeIndividSpeeds(individ, *args, **kw)
            profits[idx] = self.dataset_man.computeIndividProfit(individ["kp"])
            weights[idx] = self.dataset_man.computeIndividWeight(individ["kp"])
            tmp_distances  = self.dataset_man.individCityDistance(individ["tsp"])
            times[idx]     = (tmp_distances / speeds).sum()
            distances[idx] = tmp_distances.sum()

        # number city
        number_city = self.dataset_man.computeNumberCities(genomics.chromosomes("tsp"))
        number_obj  = self.dataset_man.computeNbrObj(genomics.chromosomes("kp"))

        CAPACITY = kw.get("W")
        if (CAPACITY < weights.min()):
            weights  = CAPACITY / weights
        else:
            weights  = CAPACITY / (weights + 1e-7)
            mask     = weights > 1
            weights[mask] = 1./weights[mask]

        # pack metrick values
        metric_values = {
            "profits"    : profits,
            "times"      : times,
            "weights"    : weights,
            "number_city": number_city,
            "number_obj" : number_obj,
            "distances"  : distances
        }
        return metric_values
    # Liniar =================================
    
    # Calculate score ++++++++++++++++++++++++++++++
    def getScoreLiniar(self, genomics, fitness_values):
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
        cities_distances = self.dataset_man.individCityDistance(individ["tsp"])
        # unpack datasets
        map_distance, item_profit, item_weight = self.dataset_man.getTupleDataset()
        # pack args
        args   = [map_distance, item_weight]
        speeds = self.__computeIndividSpeeds(individ, *args, **self._configs)
        time   = (cities_distances / speeds).sum()
        score  = profit - self._configs.get("R")*time
        return {"score": score, 
        "profit":profit, 
        "distance":distance, 
        "time":time, 
        "weight":weight, 
        "best_fitness":best_fitness}

    # Calculate score ==============================

    # utils ++++++++++++++++++++++++++++++++++++++++++++++

    # utils ==============================================
