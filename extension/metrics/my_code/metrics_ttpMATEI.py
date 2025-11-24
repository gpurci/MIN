#!/usr/bin/python

import numpy as np
from extension.metrics.my_code.metrics_base import *


class MetricsTTPMATEI(MetricsBase):
    """
    MetricsTTP:
      - computes TTP metrics for the GA (profits, times, weights, etc.)
      - compatible with the project-1 GA + the new TTPFitness we fixed.

    Methods:
      - "TTP_linear"      -> metricsTTPLiniar
      - "TTP_ada_linear"  -> metricsTTPAdaLiniar
      - "TTP_exp"         -> metricsTTPExp
    """

    def __init__(self, method, dataset, **configs):
        # CHANGED: keep MetricsBase wiring, but be explicit about name.
        super().__init__(method, name="MetricsTTP", **configs)

        # CHANGED: use the manager-style unpack to bind method -> function + getScore
        self.__fn, self.getScore = self._unpackMethod(
            method,
            TTP_linear=(self.metricsTTPLiniar, self.getScoreTTP),
            TTP_ada_linear=(self.metricsTTPAdaLiniar, self.getScoreTTP),
            TTP_exp=(self.metricsTTPExp, self.getScoreTTP),
        )

        self.dataset = dataset

    # --------------------------------------------------------------
    def __call__(self, genomics):
        # CHANGED: keep behaviour but make explicit that we pass configs
        # coming from GA / constructor.
        return self.__fn(genomics, **self._configs)

    def help(self):
        print(
            """MetricsTTP:
    metoda: 'TTP_linear';     config: -> v_min, v_max, W, alpha
    metoda: 'TTP_ada_linear'; config: -> v_min, v_max, W, alpha
    metoda: 'TTP_exp';        config: -> v_min, v_max, W, lam
"""
        )

    # ==============================================================
    #                    INDIVID METRICS
    # ==============================================================

    def computeIndividProfitKP(self, kp_individ):
        return (self.dataset["item_profit"] * kp_individ).sum()

    def computeIndividWeightKP(self, kp_individ):
        return (self.dataset["item_weight"] * kp_individ).sum()

    def computeIndividNbrObjKP(self, kp_individ):
        return kp_individ.sum()

    def computeIndividNumberCities(self, individ):
        # number of distinct cities in route
        return np.unique(
            individ, return_index=False, return_inverse=False, return_counts=False, axis=None
        ).shape[0]

    def computeIndividDistance(self, individ):
        """
        Distance of a TSP tour (cycle).
        """
        dist = self.dataset["distance"]
        d = dist[individ[:-1], individ[1:]].sum()
        d += dist[individ[-1], individ[0]]
        return d

    # ==============================================================
    #                    POPULATION METRICS
    # ==============================================================

    def computeDistances(self, population):
        return np.apply_along_axis(self.computeIndividDistance, 1, population)

    def computeNumberCities(self, population):
        return np.apply_along_axis(self.computeIndividNumberCities, 1, population)

    def computeProfitKP(self, kp_population):
        return np.apply_along_axis(self.computeIndividProfitKP, 1, kp_population)

    def computeWeightKP(self, kp_population):
        return np.apply_along_axis(self.computeIndividWeightKP, 1, kp_population)

    def computeNbrObjKP(self, kp_population):
        return np.apply_along_axis(self.computeIndividNbrObjKP, 1, kp_population)

    # ==============================================================
    #                    TTP LINEAR METRIC
    # ==============================================================

    def __computeIndividLiniarTTP(
        self, individ, *args, v_min=0.1, v_max=1.0, W=2000, alpha=0.01
    ):
        """
        Linear TTP model:
          - profit penalized by current time (alpha * Tcur)
          - speed decreases linearly as bag weight increases.
        """
        Wcur = 0.0
        Tcur = 0.0
        Pcur = 0.0

        tsp_individ = individ["tsp"]
        kp_individ = individ["kp"]

        distance, item_profit, item_weight = args

        for i in range(self.GENOME_LENGTH - 1):
            city = tsp_individ[i]
            take = kp_individ[city]

            profit = item_profit[city] * take
            weight = item_weight[city] * take

            # CHANGED: same as old code, but clarified in comment.
            Pcur += max(0.0, profit - alpha * Tcur)
            Wcur += weight

            v = v_max - v_min * (Wcur / float(W))
            v = max(v_min, v)

            Tcur += distance[city, tsp_individ[i + 1]] / v

        # return to start city
        Tcur += distance[tsp_individ[-1], tsp_individ[0]] / v
        return Pcur, Tcur, Wcur

    def metricsTTPLiniar(self, genomics, **kw):
        """
        Compute TTP-linear metrics for the whole population.
        Returns dict with:
            'profits', 'times', 'weights', 'number_city'
        (no 'number_obj' here, but TTPFitness can handle that being absent).
        """
        distance = self.dataset["distance"]
        item_profit = self.dataset["item_profit"]
        item_weight = self.dataset["item_weight"]
        args = [distance, item_profit, item_weight]

        N = genomics.shape[0]
        profits = np.zeros(N, dtype=np.float32)
        weights = np.zeros(N, dtype=np.float32)
        times = np.zeros(N, dtype=np.float32)

        for idx, individ in enumerate(genomics.population(), 0):
            profit, time, weight = self.__computeIndividLiniarTTP(
                individ, *args, **kw
            )
            profits[idx] = profit
            weights[idx] = weight
            times[idx] = time

        tsp_pop = genomics.chromosomes("tsp")
        number_city = self.computeNumberCities(tsp_pop)

        metric_values = {
            "profits": profits,
            "times": times,
            "weights": weights,
            "number_city": number_city,
        }
        return metric_values

    # ==============================================================
    #                 TTP ADA-LINEAR METRIC
    # ==============================================================

    def __computeIndividAdaLiniarTTP(
        self, individ, *args, v_min=0.1, v_max=1.0, W=2000, alpha=0.01
    ):
        """
        Adaptive-linear TTP model:
          - profit term ~ profit^2 / weight  (if weight > 0)
          - stronger time penalty (alpha * Tcur)
          - slightly different speed function.
        """
        Wcur = 0.0
        Tcur = 0.0
        Pcur = 0.0

        tsp_individ = individ["tsp"]
        kp_individ = individ["kp"]

        distance, item_profit, item_weight = args

        for i in range(self.GENOME_LENGTH - 1):
            city = tsp_individ[i]
            take = kp_individ[city]

            profit = item_profit[city] * take
            weight = item_weight[city] * take

            # CHANGED: ada-linear profit term (old project-2 idea).
            Pcur += max(0.0, profit ** 2 / (weight + 1e-7) - alpha * Tcur)
            Wcur += weight

            # CHANGED: ada-linear speed; same as in your old file.
            v = v_max - v_min * ((Wcur / float(W)) - 1.0)
            v = max(v_min, v)

            Tcur += distance[city, tsp_individ[i + 1]] / v

        Tcur += distance[tsp_individ[-1], tsp_individ[0]] / v
        return Pcur, Tcur, Wcur

    def metricsTTPAdaLiniar(self, genomics, **kw):
        """
        Adaptive-linear metrics:
          - returns same keys as TTP_linear + 'number_obj'
          - 'number_obj' is a smoothed proxy that favours reasonable
            weight usage vs. capacity and good knapsack profit.
        """
        distance = self.dataset["distance"]
        item_profit = self.dataset["item_profit"]
        item_weight = self.dataset["item_weight"]
        args = [distance, item_profit, item_weight]

        N = self.POPULATION_SIZE
        profits = np.zeros(N, dtype=np.float32)
        weights = np.zeros(N, dtype=np.float32)
        times = np.zeros(N, dtype=np.float32)

        for idx, individ in enumerate(genomics.population(), 0):
            profit, time, weight = self.__computeIndividAdaLiniarTTP(
                individ, *args, **kw
            )
            profits[idx] = profit
            weights[idx] = weight
            times[idx] = time

        # CHANGED: use chromozomes(...) instead of chromosomes(...)
        tsp_pop = genomics.chromosomes("tsp")
        kp_pop = genomics.chromosomes("kp")

        number_city = self.computeNumberCities(tsp_pop)
        tmp_profits = self.computeProfitKP(kp_pop)

        # Build "number_obj" as a soft factor depending on W and weights
        Wcap = kw.get("W", self._configs.get("W", 2000.0))

        if Wcap < weights.min():
            number_obj = np.mean(weights) / (weights + 1e-7)
        else:
            number_obj = Wcap / (weights + 1e-7)
            mask = number_obj > 1.0
            number_obj[mask] = 1.0 / number_obj[mask]

        if number_obj.max() < 10.0:
            number_obj = number_obj ** 5

        tmp_profits = tmp_profits / (tmp_profits.max() + 1e-7)
        number_obj *= tmp_profits

        metric_values = {
            "profits": profits,
            "times": times,
            "weights": weights,
            "number_city": number_city,
            "number_obj": number_obj,
        }
        return metric_values

    # ==============================================================
    #                    TTP EXPONENTIAL METRIC
    # ==============================================================

    def __computeIndividExpTTP(
        self, individ, *args, v_min=0.1, v_max=1.0, W=2000, lam=0.01
    ):
        """
        Exponential TTP model:
          - profit weighted by exp(-lam * Tcur)
        """
        Wcur = 0.0
        Tcur = 0.0
        Pcur = 0.0

        tsp_individ = individ["tsp"]
        kp_individ = individ["kp"]

        distance, item_profit, item_weight = args

        for i in range(self.GENOME_LENGTH - 1):
            city = tsp_individ[i]
            take = kp_individ[city]

            profit = item_profit[city] * take
            weight = item_weight[city] * take

            Pcur += profit * np.exp(-lam * Tcur)
            Wcur += weight

            v = v_max - (v_max - v_min) * (Wcur / float(W))
            Tcur += distance[city, tsp_individ[i + 1]] / v

        v = v_max - (v_max - v_min) * (Wcur / float(W))
        Tcur += distance[tsp_individ[-1], tsp_individ[0]] / v
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

        for idx, individ in enumerate(genomics.population(), 0):
            profit, time, weight = self.__computeIndividExpTTP(
                individ, *args, **kw
            )
            profits[idx] = profit
            weights[idx] = weight
            times[idx] = time

        tsp_pop = genomics.chromosomes("tsp")
        kp_pop = genomics.chromosomes("kp")
        number_city = self.computeNumberCities(tsp_pop)
        number_obj = self.computeNbrObjKP(kp_pop)

        # CHANGED: reuse Wcap-based scaling like in Ada-linear variant
        Wcap = kw.get("W", self._configs.get("W", 2000.0))
        number_obj = number_obj * Wcap / (weights * self.GENOME_LENGTH + 1e-7)

        mask = Wcap <= weights
        number_obj[mask] = 1.0

        return {
            "profits": profits,
            "times": times,
            "weights": weights,
            "number_city": number_city,
            "number_obj": number_obj,
        }

    # =============================================================
    #                    UTILITY AND SCORE
    # =============================================================

    def computeSpeedTTP(self, Wcur, v_max, v_min, Wmax):
        """
        Standard TTP speed: v = v_max - (v_max - v_min)*(Wcur / Wmax),
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
        Distance of a TSP tour.
        """
        if distance is None:
            distance = self.dataset["distance"]
        d = distance[tsp_individ[:-1], tsp_individ[1:]].sum()
        d += distance[tsp_individ[-1], tsp_individ[0]]
        return d

    def getScoreTTP(self, genomics, fitness_values):
        """
        Logging score for TTP:

        - score   := standard TTP score = profit - R * time
        - profit  := raw knapsack profit (sum of picked item profits)
        - time    := total travel time along the tour (TTP model)
        - distance: geometric TSP tour length
        - weight  := total knapsack weight
        - best_fitness := value from TTPFitness
        """
        # index of best individual according to fitness
        arg_best = self.getArgBest(fitness_values)
        individ = genomics[arg_best]
        best_fitness = float(fitness_values[arg_best])

        # remember best in GA
        genomics.setBest(individ)

        # ----- basic metrics -----
        tsp_individ = individ["tsp"]
        kp_individ  = individ["kp"]

        distance = self.computeIndividDistance(tsp_individ)
        profit   = float(self.computeIndividProfitKP(kp_individ))
        weight   = float(self.computeIndividWeightKP(kp_individ))

        # ----- parameters for time & score -----
        v_min = float(self._configs.get("v_min", 0.1))
        v_max = float(self._configs.get("v_max", 1.0))
        Wmax  = float(self._configs.get("W",   2000.0))

        # Renting rate R: try configs, then dataset, then fallback
        if "R" in self._configs:
            R = float(self._configs["R"])
        elif isinstance(getattr(self, "dataset", None), dict) and "R" in self.dataset:
            R = float(self.dataset["R"])
        else:
            R = 1.0  # fallback

        dist_mat    = self.dataset["distance"]
        item_weight = self.dataset["item_weight"]

        # ----- compute standard TTP travel time -----
        Wcur = 0.0
        Tcur = 0.0

        # traverse tour edges
        for i in range(len(tsp_individ) - 1):
            city = tsp_individ[i]
            take = kp_individ[city]
            Wcur += item_weight[city] * take

            v = self.computeSpeedTTP(Wcur, v_max, v_min, Wmax)
            Tcur += dist_mat[city, tsp_individ[i + 1]] / v

        # edge from last city back to start
        v = self.computeSpeedTTP(Wcur, v_max, v_min, Wmax)
        Tcur += dist_mat[tsp_individ[-1], tsp_individ[0]] / v

        # ----- standard TTP score -----
        ttp_score = profit - R * Tcur

        return {
            "score":        ttp_score,   # profit - R * time
            "profit":       profit,      # raw profit
            "time":         Tcur,        # <-- HERE
            "distance":     distance,
            "weight":       weight,
            "best_fitness": best_fitness,
        }



# OLD helper kept for backwards compatibility with any legacy code
def normalization(x):
    x = np.asarray(x, dtype=np.float32)
    x_min = x.min()
    x_max = x.max()
    denom = x_max - x_min
    if denom == 0:
        return np.ones_like(x, dtype=np.float32)
    return (x_max - x) / denom
