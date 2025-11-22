#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import *

class DefaultMetricsTTP(RootGA):
    """
    Extension metrics ONLY for TTP.
    Contains manager Metrics TTP methods unchanged.
    """

    def __init__(self, method=None, **kw):
        super().__init__()
        self.__configs = kw
        self.__method = method
        self.__fn = self.__unpackMethod(method)

    def __str__(self):
        return f"DefaultMetricsTTP(method={self.__method}, configs={self.__configs})"

    def help(self):
        return """DefaultMetricsTTP:
    metoda: 'TTP_linear'     config -> v_min, v_max, W, alpha
    metoda: 'TTP_ada_linear' config -> v_min, v_max, W, alpha
    metoda: 'TTP_exp'        config -> v_min, v_max, W, lam
"""

    def __unpackMethod(self, method):
        fn = self.metricsAbstract
        if method == "TTP_linear":
            fn = self.metricsTTPLiniar
            self.getScore = self.getScoreTTP
        elif method == "TTP_ada_linear":
            fn = self.metricsTTPAdaLiniar
            self.getScore = self.getScoreTTP
        elif method == "TTP_exp":
            fn = self.metricsTTPExp
            self.getScore = self.getScoreTTP
        return fn

    def __call__(self, genomics, **call_configs):
        cfg = self.__configs.copy()
        cfg.update(call_configs)
        return self.__fn(genomics, **cfg)

    def setParameters(self, **kw):
        super().setParameters(**kw)

    def setDataset(self, dataset):
        self.dataset = dataset

    def getDataset(self):
        return self.dataset

    def metricsAbstract(self, genomics, **kw):
        raise NameError(
            f"Lipseste metoda '{self.__method}' pentru DefaultMetricsTTP: config '{self.__configs}'"
        )
    def __getIndividNumberCities(self, individ):
        mask_cities = np.zeros(self.GENOME_LENGTH, dtype=np.int32)
        mask_cities[individ] = 1
        return mask_cities.sum()

    def __getNumberCities(self, population):
        number_citys = []
        for individ in population:
            number_citys.append(self.__getIndividNumberCities(individ))
        return np.array(number_citys, dtype=np.float32)

    def computeIndividProfitKP(self, kp_individ, profits):
        return (kp_individ * profits).sum()

    def computeIndividWeightKP(self, kp_individ, weights):
        return (kp_individ * weights).sum()

    def computeIndividNbrObjKP(self, kp_individ):
        return kp_individ.sum()

    def computeNbrObjKP(self, kp_population):
        return np.apply_along_axis(self.computeIndividNbrObjKP, axis=1, arr=kp_population)

    # ---------- TTP Linear ----------
    def __computeIndividLiniarTTP(self, individ, *args,
                                  v_min=0.1, v_max=1, W=2000, alpha=0.01):
        Wcur = 0.0
        Tcur = 0.0
        Pcur = 0.0

        tsp_individ = individ["tsp"]
        kp_individ  = individ["kp"]

        distance, item_profit, item_weight = args

        for i in range(self.GENOME_LENGTH - 1):
            city = tsp_individ[i]
            take = kp_individ[city]

            profit = item_profit[city] * take
            weight = item_weight[city] * take

            Pcur += max(0.0, profit - alpha * Tcur)
            Wcur += weight

            v = v_max - v_min * (Wcur / W)
            v = max(v_min, v)

            Tcur += distance[city, tsp_individ[i + 1]] / v

        Tcur += distance[tsp_individ[-1], tsp_individ[0]] / v
        return Pcur, Tcur, Wcur

    def metricsTTPLiniar(self, genomics, **kw):
        distance    = self.dataset["distance"]
        item_profit = self.dataset["item_profit"]
        item_weight = self.dataset["item_weight"]
        args = [distance, item_profit, item_weight]

        profits = np.zeros(self.POPULATION_SIZE, dtype=np.float32)
        weights = np.zeros(self.POPULATION_SIZE, dtype=np.float32)
        times   = np.zeros(self.POPULATION_SIZE, dtype=np.float32)

        for idx, individ in enumerate(genomics.population(), 0):
            profit, time, weight = self.__computeIndividLiniarTTP(individ, *args, **kw)
            profits[idx] = profit
            weights[idx] = weight
            times[idx]   = time

        number_city = self.__getNumberCities(genomics.chromozomes("tsp"))

        return {
            "profits": profits,
            "times": times,
            "weights": weights,
            "number_city": number_city
        }

    # ---------- TTP Adaptive Linear ----------
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
        distance    = self.dataset["distance"]
        item_profit = self.dataset["item_profit"]
        item_weight = self.dataset["item_weight"]
        args = [distance, item_profit, item_weight]

        profits = np.zeros(self.POPULATION_SIZE, dtype=np.float32)
        weights = np.zeros(self.POPULATION_SIZE, dtype=np.float32)
        times   = np.zeros(self.POPULATION_SIZE, dtype=np.float32)

        for idx, individ in enumerate(genomics.population(), 0):
            profit, time, weight = self.__computeIndividAdaLiniarTTP(individ, *args, **kw)
            profits[idx] = profit
            weights[idx] = weight
            times[idx]   = time

        number_city = self.__getNumberCities(genomics.chromozomes("tsp"))
        number_obj  = self.computeNbrObjKP(genomics.chromozomes("kp"))
        number_obj  = number_obj * kw.get("W") / (weights * self.GENOME_LENGTH + 1e-7)

        mask = (kw.get("W") <= weights)
        number_obj[mask] = 1.0

        return {
            "profits": profits,
            "times": times,
            "weights": weights,
            "number_city": number_city,
            "number_obj": number_obj
        }

    # ---------- TTP Exponential ----------
    def __computeIndividExpTTP(self, individ, *args,
                               v_min=0.1, v_max=1, W=2000, lam=0.01):
        Wcur = 0.0
        Tcur = 0.0
        Pcur = 0.0

        tsp_individ = individ["tsp"]
        kp_individ  = individ["kp"]

        distance, item_profit, item_weight = args

        for i in range(self.GENOME_LENGTH - 1):
            city = tsp_individ[i]
            take = kp_individ[city]

            profit = item_profit[city] * take
            weight = item_weight[city] * take

            Pcur += profit * np.exp(-lam * Tcur)
            Wcur += weight

            v = v_max - (v_max - v_min) * (Wcur / W)
            Tcur += distance[city, tsp_individ[i + 1]] / v

        v = v_max - (v_max - v_min) * (Wcur / W)
        Tcur += distance[tsp_individ[-1], tsp_individ[0]] / v
        return Pcur, Tcur, Wcur

    def metricsTTPExp(self, genomics, **kw):
        distance    = self.dataset["distance"]
        item_profit = self.dataset["item_profit"]
        item_weight = self.dataset["item_weight"]
        args = [distance, item_profit, item_weight]

        profits = np.zeros(self.POPULATION_SIZE, dtype=np.float32)
        weights = np.zeros(self.POPULATION_SIZE, dtype=np.float32)
        times   = np.zeros(self.POPULATION_SIZE, dtype=np.float32)

        for idx, individ in enumerate(genomics.population(), 0):
            profit, time, weight = self.__computeIndividExpTTP(individ, *args, **kw)
            profits[idx] = profit
            weights[idx] = weight
            times[idx]   = time

        number_city = self.__getNumberCities(genomics.chromozomes("tsp"))

        return {
            "profits": profits,
            "times": times,
            "weights": weights,
            "number_city": number_city
        }

    def getIndividDistanceTTP(self, tsp_individ, distance):
        d = distance[tsp_individ[:-1], tsp_individ[1:]]
        d = np.concatenate((d, [distance[tsp_individ[-1], tsp_individ[0]]]))
        return d.sum()

    def getScoreTTP(self, genomics, fitness_values):
        arg_best = self.getArgBest(fitness_values)
        tsp_individ = genomics.chromozomes("tsp")[arg_best]
        d = self.getIndividDistanceTTP(tsp_individ, self.dataset["distance"])
        return {"score": d, "best_fitness": fitness_values[arg_best]}

    def getArgBest(self, fitness_values):
        return np.argmax(fitness_values)

    def getBestIndivid(self):
        arg_best = self.getArgBest(self.fitness_values)
        return self.genomics[arg_best]
