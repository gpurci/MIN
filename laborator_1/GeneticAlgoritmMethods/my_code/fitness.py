#!/usr/bin/python
import numpy as np
from root_GA import *


class Fitness(RootGA):
    """
    Clasa 'Fitness'
    Oferă metode pentru a calcula funcția fitness a populației.
    """

    def __init__(self, method, **kw):
        super().__init__()
        self.__configs = kw
        self.__setMethods(method)

    # ------------------------------------------------------------------
    def __call__(self, metric_values):
        return self.fn(metric_values, **self.__configs)

    # ------------------------------------------------------------------
    def __str__(self):
        return f"Fitness: method: {self.__method} configs: {self.__configs}"

    # ------------------------------------------------------------------
    def help(self):
        return """Fitness:
    metoda: 'TSP_f1score'
    metoda: 'TSP_norm'
    metoda: 'TTP_f1score' -> config: R=1
    metoda: 'TTP' -> config: R=1
"""

    # ------------------------------------------------------------------
    def __setMethods(self, method):
        self.__method = method
        self.fn = self.__unpack_method(method)

    # ------------------------------------------------------------------
    def __unpack_method(self, method):

        fn = self.fitnessAbstract

        if method == "TSP_f1score":
            fn = self.fitnessF1scoreTSP
        elif method == "TSP_norm":
            fn = self.fitnessNormTSP
        elif method == "TTP_f1score":
            fn = self.fitnessF1scoreTTP
        elif method == "TTP":
            fn = self.fitnessTTP

        return fn

    # ------------------------------------------------------------------
    def fitnessAbstract(self, metric_values: dict):
        raise NameError(
            f"Lipseste metoda '{self.__method}' pentru functia Fitness, configs={self.__configs}"
        )

    # ==================================================================
    #                   FITNESS — TSP F1 SCORE
    # ==================================================================
    def fitnessF1scoreTSP(self, metric_values: dict):
        """
        Returnează F1 pentru TSP:
            2 * dist_norm * city_norm / (dist_norm + city_norm)
        """

        distances = metric_values["distances"]
        number_city = metric_values["number_city"]

        # normalize
        number_city = self.__cityBinaryTSP(number_city)
        distances = self.__distanceF1scoreTSP(distances)

        fitness_values = (2 * distances * number_city) / (
            distances + number_city + 1e-7
        )

        return fitness_values

    # ------------------------------------------------------------------
    def __distanceF1scoreTSP(self, distances):

        mask_not_zero = distances != 0
        valid = distances[mask_not_zero]

        min_distance = valid.min() if valid.size > 0 else 0.1

        distances[:] = 0.1
        return (2 * min_distance) / (distances + min_distance)

    # ------------------------------------------------------------------
    def __cityBinaryTSP(self, number_city):
        return (number_city >= (self.GENOME_LENGTH - 1)).astype(np.float32)

    # ==================================================================
    #                        FITNESS — TSP NORM
    # ==================================================================
    def fitnessNormTSP(self, metric_values: dict):

        distances = metric_values["distances"]
        number_city = metric_values["number_city"]

        number_city = self.__cityNormTSP(number_city)
        distances = self.__norm(distances)

        return (2 * distances * number_city) / (distances + number_city + 1e-7)

    # ------------------------------------------------------------------
    def __norm(self, x):
        x_min = x.min()
        x_max = x.max()
        return (x_max - x) / (x_max - x_min)

    # ------------------------------------------------------------------
    def __cityNormTSP(self, number_city):
        mask = (number_city >= (self.GENOME_LENGTH - 5)).astype(np.float32)
        return mask * (number_city / self.GENOME_LENGTH) ** 5

    # ==================================================================
    #                           FITNESS — TTP
    # ==================================================================
    def fitnessTTP(self, metric_values, R=1):
        """
        Fitness direct din TTP:
            score = profit - R * time

        Normalizat la > 0.
        """

        scores = metric_values.get("score", None)
        if scores is None:
            raise KeyError("metric_values must contain key 'score' for TTP fitness")

        # shift to positive
        min_score = scores.min()
        if min_score <= 0:
            scores = scores - min_score + 1e-6

        # normalize 0..1
        max_score = scores.max()
        return scores / max_score if max_score > 0 else scores

    # ==================================================================
    #                        FITNESS — TTP F1 SCORE
    # ==================================================================
    def fitnessF1scoreTTP(self, metric_values, R=1, beta=1.0, W=None):
        """
        F1-like pentru TTP, cu penalizare soft pe depășirea capacității.

        raw = 2 * profit / (profit + R*time + penalty)
        """

        profits = metric_values["profits"]
        times = metric_values["times"]
        weights = metric_values["weights"]
        number_city = metric_values["number_city"]

        # trebuie rută completă
        mask_city = self.__cityBinaryTSP(number_city)

        # determină Wmax
        if W is None:
            Wmax = self.__configs.get("W", None)
        else:
            Wmax = W

        if Wmax is None:
            raise ValueError("Trebuie fitness={'method':'TTP_f1score', 'W':25936}")

        # penalizare depășire
        overweight = np.maximum(0.0, weights - Wmax)
        penalty = beta * overweight

        # scor F1-like
        raw = mask_city * (
            2.0 * profits / (profits + R * times + penalty + 1e-9)
        )

        # shift la > 0
        min_raw = raw.min()
        if min_raw <= 0:
            raw = raw - min_raw + 1e-6

        return raw
