#!/usr/bin/python
import numpy as np
from root_GA import *


class Metrics(RootGA):
    """
    Clasa 'Metrics', ofera metode pentru a calcula metrici pentru probleme
    TSP si TTP. Functia '__call__' aplica metrica configurata.
    """

    def __init__(self, method, **kw):
        super().__init__()
        self.__configs = kw
        self.__setMethods(method)

    def __str__(self):
        return f"Metrics: method: {self.__method} configs: {self.__configs}"

    def __call__(self, genomics):
        return self.fn(genomics, **self.__configs)

    def help(self):
        return (
            "Metrics:\n"
            "  metoda: 'TTP_linear'; config: v_min, v_max, W, alpha\n"
            "  metoda: 'TTP_mean_linear'; config: v_min, v_max, W, alpha\n"
            "  metoda: 'TTP_exp'; config: v_min, v_max, W, lam\n"
            "  metoda: 'TSP'; config: None\n"
        )

    # =============================================================
    # METHOD SELECTION
    # =============================================================
    def __unpack_method(self, method):
        fn = self.metricsAbstract

        if method is not None:
            if method == "TSP":
                fn = self.metricsTSP
                self.getScore = self.getScoreTSP

            elif method == "TTP_linear":
                fn = self.metricsTTPLiniar
                self.getScore = self.getScoreTTP

            elif method == "TTP_mean_linear":
                fn = self.metricsTTPMeanLiniar
                self.getScore = self.getScoreTTP

            elif method == "TTP_exp":
                fn = self.metricsTTPExp
                self.getScore = self.getScoreTTP

        return fn

    def __setMethods(self, method):
        self.__method = method
        self.fn = self.__unpack_method(method)

    # =============================================================
    # DATASET MANAGEMENT
    # =============================================================
    def setDataset(self, dataset):
        print(
            f"Utilizezi metoda: {self.__method}, datele trebuie sa corespunda "
            f"metodei de calcul a metricilor!!!"
        )
        self.dataset = dataset

    def getDataset(self):
        return self.dataset

    # =============================================================
    # BEST INDIVIDUAL
    # =============================================================
    def getArgBest(self, fitness_values):
        return np.argmax(fitness_values)

    def getBestIndivid(self):
        return self.__best_individ

    # =============================================================
    # DEFAULT METRIC
    # =============================================================
    def metricsAbstract(self, population):
        raise NameError(
            f"Lipseste metoda '{self.__method}' pentru functia de 'Metrics': "
            f"config '{self.__configs}'"
        )

    # =============================================================
    #                 TSP METRICS
    # =============================================================
    def __getIndividDistance(self, individ):
        distances = self.dataset["distance"][individ[:-1], individ[1:]]
        return distances.sum() + self.dataset["distance"][individ[-1], individ[0]]

    def __getIndividNumberCities(self, individ):
        return np.unique(individ[:-1]).shape[0]

    def __getDistances(self, population):
        return np.apply_along_axis(self.__getIndividDistance, 1, population)

    def __getNumberCities(self, population):
        return np.apply_along_axis(self.__getIndividNumberCities, 1, population)

    def metricsTSP(self, genomics):
        population = genomics.chromozomes("tsp")

        distances = self.__getDistances(population)
        number_city = self.__getNumberCities(population)

        return {
            "distances": distances,
            "number_city": number_city
        }

    def getScoreTSP(self, genomics, fitness_values):
        arg_best = self.getArgBest(fitness_values)
        individ = genomics[arg_best]["tsp"]

        score = self.__getIndividDistance(individ)
        best_fitness = fitness_values[arg_best]

        self.__best_individ = individ
        return {"score": score, "best_fitness": best_fitness}

    # =============================================================
    #                 TTP – PROFIT / WEIGHT
    # =============================================================
    def computeIndividProfitKP(self, kp_individ):
        return (self.dataset["item_profit"] * kp_individ).sum()

    def computeIndividWeightKP(self, kp_individ):
        return (self.dataset["item_weight"] * kp_individ).sum()

    def computeProfitKP(self, kp_population):
        return np.apply_along_axis(self.computeIndividProfitKP, 1, kp_population)

    def computeWeightKP(self, kp_population):
        return np.apply_along_axis(self.computeIndividWeightKP, 1, kp_population)

    # =============================================================
    #                 TTP – LINEAR MODEL
    # =============================================================
    def __computeIndividLiniarTTP(
        self, individ, v_min=0.1, v_max=1.0, W=25936, alpha=0.0, R=5.61
    ):
        tsp = individ["tsp"]
        kp = individ["kp"]

        Pcur = 0.0
        Wcur = 0.0
        Tcur = 0.0

        distance = self.dataset["distance"]
        item_profit = self.dataset["item_profit"]
        item_weight = self.dataset["item_weight"]

        for i in range(len(tsp) - 1):
            city = tsp[i]

            # one item per city
            if city > 0 and kp[city] == 1:
                item_idx = city - 1
                profit = item_profit[item_idx]
                weight = item_weight[item_idx]

                Pcur += max(0.0, profit - alpha * Tcur)
                Wcur += weight

            v = v_max - (v_max - v_min) * (Wcur / W)
            v = max(v_min, v)

            nxt = tsp[i + 1]
            Tcur += distance[city, nxt] / v

        # return to start
        last = tsp[-1]
        start = tsp[0]
        v = v_max - (v_max - v_min) * (Wcur / W)
        v = max(v_min, v)
        Tcur += distance[last, start] / v

        score = Pcur - R * Tcur
        return score, Pcur, Tcur, Wcur

    def metricsTTPLiniar(self, genomics, **kw):
        N = self.POPULATION_SIZE
        scores = np.zeros(N)
        profits = np.zeros(N)
        times = np.zeros(N)
        weights = np.zeros(N)

        for i, individ in enumerate(genomics.population()):
            score, P, T, W = self.__computeIndividLiniarTTP(
                individ,
                v_min=kw.get("v_min", 0.1),
                v_max=kw.get("v_max", 1.0),
                W=kw.get("W", 25936),
                alpha=kw.get("alpha", 0.0),
                R=kw.get("R", 5.61)
            )
            scores[i] = score
            profits[i] = P
            times[i] = T
            weights[i] = W

        number_city = self.__getNumberCities(genomics.chromozomes("tsp"))

        return {
            "score": scores,
            "profits": profits,
            "times": times,
            "weights": weights,
            "number_city": number_city
        }

    # =============================================================
    #                 TTP – MEAN LINEAR MODEL
    # =============================================================
    def __computeIndividLiniarMeanTTP(
        self, individ, *args, v_min=0.1, v_max=1, W=2000,
        CAPACITY=0, alpha=0.01
    ):
        Wcur = 0.0
        Tcur = 0.0
        Pcur = 0.0

        tsp = individ["tsp"]
        kp = individ["kp"]

        distance, item_profit, item_weight = args

        for i in range(self.GENOME_LENGTH - 1):
            city = tsp[i]

            take = kp[city]
            profit = item_profit[city] * take
            weight = item_weight[city] * take

            Pcur += max(0.0, profit - alpha * Tcur)
            Wcur += weight

            v = v_max - (v_max - v_min) * (Wcur / CAPACITY)
            v = max(v_min, v)

            Tcur += distance[city, tsp[i + 1]] / v

        v = v_max - (v_max - v_min) * (Wcur / CAPACITY)
        Tcur += distance[tsp[-1], tsp[0]] / v

        return Pcur, Tcur, Wcur

    def metricsTTPMeanLiniar(self, genomics, **kw):
        distance = self.dataset["distance"]
        item_profit = self.dataset["item_profit"]
        item_weight = self.dataset["item_weight"]

        weights = self.computeWeightKP(genomics.chromozomes("kp"))
        min_weight = weights.min()
        W = kw.get("W", 20000)

        CAPACITY = weights.mean() if min_weight > W else W

        args = [distance, item_profit, item_weight]

        N = self.POPULATION_SIZE
        profits = np.zeros(N, dtype=np.float32)
        times = np.zeros(N, dtype=np.float32)
        weights = np.zeros(N, dtype=np.float32)

        for i, individ in enumerate(genomics.population()):
            profit, time, weight = self.__computeIndividLiniarMeanTTP(
                individ, *args, CAPACITY=CAPACITY, **kw
            )
            profits[i] = profit
            times[i] = time
            weights[i] = weight

        number_city = self.__getNumberCities(genomics.chromozomes("tsp"))

        return {
            "profits": profits,
            "times": times,
            "weights": weights,
            "number_city": number_city
        }

    # =============================================================
    #                 TTP – EXPONENTIAL MODEL
    # =============================================================
    def __computeIndividExpTTP(
        self, individ, *args, v_min=0.1, v_max=1, W=2000, lam=0.01
    ):
        Wcur = 0.0
        Tcur = 0.0
        Pcur = 0.0

        tsp = individ["tsp"]
        kp = individ["kp"]

        distance, item_profit, item_weight = args

        for i in range(self.GENOME_LENGTH - 1):
            city = tsp[i]
            take = kp[city]

            profit = item_profit[city] * take
            weight = item_weight[city] * take

            Pcur += profit * np.exp(-lam * Tcur)
            Wcur += weight

            v = v_max - (v_max - v_min) * (Wcur / W)
            Tcur += distance[city, tsp[i + 1]] / v

        v = v_max - (v_max - v_min) * (Wcur / W)
        Tcur += distance[tsp[-1], tsp[0]] / v

        return Pcur, Tcur, Wcur

    def metricsTTPExp(self, genomics, **kw):
        distance = self.dataset["distance"]
        item_profit = self.dataset["item_profit"]
        item_weight = self.dataset["item_weight"]

        args = [distance, item_profit, item_weight]

        N = self.POPULATION_SIZE
        profits = np.zeros(N, dtype=np.float32)
        weights = np.zeros(N, dtype=np.float32)
        times = np.zeros(N, dtype=np.float32)

        for i, individ in enumerate(genomics.population()):
            profit, time, weight = self.__computeIndividLiniarTTP(
                individ, *args, **kw
            )
            profits[i] = profit
            weights[i] = weight
            times[i] = time

        return {
            "profits": profits,
            "weights": weights,
            "times": times
        }

    # =============================================================
    #                 TTP SCORE SUMMARY
    # =============================================================
    def getScoreTTP(self, genomics, fitness_values):
        """Returnează scorul real TTP = profit - R * time."""

        # Best individual
        arg_best = self.getArgBest(fitness_values)
        individ = genomics[arg_best]
        best_fitness = fitness_values[arg_best]

        tsp_individ = individ["tsp"]
        kp_individ = individ["kp"]

        # baseline stats
        distance = self.__getIndividDistance(tsp_individ)
        profit = self.computeIndividProfitKP(kp_individ)
        weight = self.computeIndividWeightKP(kp_individ)

        # extract configs
        R = self.__configs.get("R", 5.61)
        v_min = self.__configs.get("v_min", 0.1)
        v_max = self.__configs.get("v_max", 1.0)
        Wcap = self.__configs.get("W", 25936)
        alpha = self.__configs.get("alpha", 0.01)

        # compute time using the same TTP-linear evaluator (4 returned values!)
        score_lin, Pcur_lin, time, Wcur_lin = self._Metrics__computeIndividLiniarTTP(
            individ,
            v_min=v_min,
            v_max=v_max,
            W=Wcap,
            alpha=alpha,
            R=R
        )

        # true TTP objective
        score = profit - R * time

        self.__best_individ = individ

        return {
            "score": score,
            "profit": profit,
            "distance": distance,
            "time": time,
            "weight": weight,
            "best_fitness": best_fitness
        }
