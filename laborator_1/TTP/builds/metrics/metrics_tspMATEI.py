#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import *

class MetricsTSP(RootGA):
    """
    Extension metrics ONLY for TSP.
    Contains manager Metrics.TSP methods unchanged.
    """

    def __init__(self, method="TSP", **kw):
        super().__init__()
        self.__configs = kw
        self.__method = method
        self.__fn = self.__unpackMethod(method)

    def __str__(self):
        return f"MetricsTSP(method={self.__method}, configs={self.__configs})"

    def help(self):
        return """MetricsTSP:
    metoda: 'TSP'; config: None;
    (extern handled by manager)
"""

    def __unpackMethod(self, method):
        fn = self.metricsAbstract
        if method == "TSP":
            fn = self.metricsTSP
            self.getScore = self.getScoreTSP
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
            f"Lipseste metoda '{self.__method}' pentru MetricsTSP: config '{self.__configs}'"
        )

    def __getIndividDistance(self, individ, args_distances):
        distances = args_distances[individ[:-1], individ[1:]]
        distance_individ = np.concatenate(
            (distances, [args_distances[individ[-1], individ[0]]]),
            axis=0
        )
        return distance_individ.sum()

    def __getIndividNumberCities(self, individ):
        mask_cities = np.zeros(self.GENOME_LENGTH, dtype=np.int32)
        mask_cities[individ] = 1
        return mask_cities.sum()

    def __getDistances(self, population, args_distances):
        distances = []
        for individ in population:
            distances.append(self.__getIndividDistance(individ, args_distances))
        return np.array(distances, dtype=np.float32)

    def __getNumberCities(self, population):
        number_citys = []
        for individ in population:
            number_citys.append(self.__getIndividNumberCities(individ))
        return np.array(number_citys, dtype=np.float32)

    def metricsTSP(self, genomics):
        args_distances = self.dataset.get("distance", None)
        if args_distances is None:
            raise Exception("Lipseste matricea de distante")

        tsp_population = genomics.chromozomes("tsp")
        if tsp_population is None:
            raise NameError("Lipseste cromozomul 'tsp'")

        distances   = self.__getDistances(tsp_population, args_distances)
        number_city = self.__getNumberCities(tsp_population)

        metric_values = {
            "distances": distances,
            "number_city": number_city
        }
        return metric_values

    def getScoreTSP(self, genomics, fitness_values):
        arg_best = self.getArgBest(fitness_values)
        tsp_best = genomics.chromozomes("tsp")[arg_best]
        score = self.__getIndividDistance(tsp_best, self.dataset["distance"])
        return {"score": score, "best_fitness": fitness_values[arg_best]}

    def getArgBest(self, fitness_values):
        return np.argmax(fitness_values)

    def getBestIndivid(self):
        arg_best = self.getArgBest(self.fitness_values)
        return self.genomics[arg_best]
