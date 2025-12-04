#!/usr/bin/python

import numpy as np
from extension.metrics.metrics_base import *
from extension.utils.normalization import *

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
        # creaza vectori de monitoring goi
        profits   = np.zeros(self.POPULATION_SIZE, dtype=np.float32)
        weights   = np.zeros(self.POPULATION_SIZE, dtype=np.float32)
        times     = np.zeros(self.POPULATION_SIZE, dtype=np.float32)
        distances = np.zeros(self.POPULATION_SIZE, dtype=np.float32)
        # unpack configs
        v_min, v_max, CAPACITY = kw.get("v_min"), kw.get("v_max"), kw.get("W")
        # calculate metrics for every individ from population
        for idx, individ in enumerate(genomics.population(), 0):
            speeds = self.dataset_man.computeIndividSpeeds(individ["tsp"], individ["kp"], 
                                            v_min=v_min, v_max=v_max, W=CAPACITY)
            profits[idx] = self.dataset_man.computeIndividProfit(individ["kp"])
            weights[idx] = self.dataset_man.computeIndividWeight(individ["kp"])
            tmp_distances  = self.dataset_man.individCityDistance(individ["tsp"])
            times[idx]     = (tmp_distances / speeds).sum()
            distances[idx] = tmp_distances.sum()

        # number city
        number_city = self.dataset_man.computeNumberCities(genomics.chromosomes("tsp"))
        number_obj  = self.dataset_man.computeNbrObj(genomics.chromosomes("kp"))

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
        # unpack
        tsp_individ = individ["tsp"]
        kp_individ  = individ["kp"]
        # unpack config
        v_min, v_max = self._configs.get("v_min"), self._configs.get("v_max")
        W,     R     = self._configs.get("W"), self._configs.get("R")
        # get best fitness
        best_fitness = fitness_values[arg_best]
        # set best individ
        genomics.setBest(individ)
        # conpute distance
        distance = self.dataset_man.computeIndividDistance(tsp_individ)
        # compute profit and weight
        profit = self.dataset_man.computeIndividProfit(kp_individ)
        weight = self.dataset_man.computeIndividWeight(kp_individ)
        # calculeaza distanta dintre orase pentru cel mai bun individ
        cities_distances = self.dataset_man.individCityDistance(tsp_individ)
        # calculeaza viteza pentru cel mai bun individ
        speeds = self.dataset_man.computeIndividSpeeds(tsp_individ, kp_individ, v_min=v_min, v_max=v_max, W=W)
        # calculeaza timpul pentru cel mai bun individ
        time   = (cities_distances / speeds).sum()
        # calculeaza scorul pentru cel mai bun individ
        score  = profit - R*time
        return {"score": score, 
            "profit":profit, 
            "distance":distance, 
            "time":time, 
            "weight":weight, 
            "best_fitness":best_fitness
        }

    # Calculate score ==============================

    # utils ++++++++++++++++++++++++++++++++++++++++++++++

    # utils ==============================================
