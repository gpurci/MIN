#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import RootGA

class MetricsTTP(RootGA):
    """
    Metrics for TTP.
    Supports:
        - TTP_standard   (classical profit/time/weight metrics)
        - TTP_linear
        - TTP_ada_linear
        - TTP_exp
    """

    def __init__(self, method, dataset, **kw):
        super().__init__()
        self.__configs = kw
        self.dataset   = dataset
        self.__setMethods(method)

        self.distance    = dataset["distance"]
        self.item_profit = dataset["item_profit"]
        self.item_weight = dataset["item_weight"]

        # will be filled by GeneticAlgorithm with current metric_values
        self.metrics_cache = None

    def __str__(self):
        return f"MetricsTTP(method={self.__method}, configs={self.__configs})"

    def __call__(self, genomics):
        mv = self.__fn(genomics, **self.__configs)
        self.metrics_cache = mv
        return mv


    def help(self):
        print("""MetricsTTP:
    method: 'TTP_standard';   configs: v_min,v_max,Wmax
    method: 'TTP_linear';     configs: v_min,v_max,W,alpha
    method: 'TTP_ada_linear'; configs: v_min,v_max,W,alpha
    method: 'TTP_exp';        configs: v_min,v_max,W,lam
""")

    # ---------------- dispatch ----------------
    def __unpackMethod(self, method):
        table = {
            "TTP_standard":   self.metricsTTPStandard,
            "TTP_linear":     self.metricsTTPLiniar,
            "TTP_ada_linear": self.metricsTTPAdaLiniar,
            "TTP_exp":        self.metricsTTPExp,
        }
        self.getScore = self.getScoreTTP
        return table.get(method, self.metricsAbstract)

    def __setMethods(self, method):
        self.__method = method
        self.__fn = self.__unpackMethod(method)

    def getArgBest(self, fitness_values):
        return int(np.argmax(fitness_values, axis=None))

    def metricsAbstract(self, genomics, **kw):
        raise NameError(
            f"Missing method '{self.__method}' for MetricsTTP, configs={self.__configs}"
        )

    # ---------------- helpers ----------------
    def computeIndividProfitKP(self, kp_individ):
        return (self.item_profit * kp_individ).sum()

    def computeIndividWeightKP(self, kp_individ):
        return (self.item_weight * kp_individ).sum()

    def computeIndividNumberCities(self, individ):
        return np.unique(individ).shape[0]

    def computeIndividDistance(self, tsp_individ):
        d = self.distance[tsp_individ[:-1], tsp_individ[1:]].sum()
        d += self.distance[tsp_individ[-1], tsp_individ[0]]
        return float(d)

    def computeDistances(self, population):
        return np.apply_along_axis(self.computeIndividDistance, axis=1, arr=population)

    def computeNumberCities(self, population):
        return np.apply_along_axis(self.computeIndividNumberCities, axis=1, arr=population)

    def computeNbrObjKP(self, kp_population):
        return np.apply_along_axis(lambda x: x.sum(), axis=1, arr=kp_population)

    def computeSpeedTTP(self, Wcur, v_max, v_min, Wmax):
        v = v_max - (v_max - v_min) * (Wcur / float(Wmax))
        return min(v_max, max(v_min, v))

    def getIndividDistanceTTP(self, tsp_individ, distance=None):
        if distance is None:
            distance = self.dataset["distance"]
        d = distance[tsp_individ[:-1], tsp_individ[1:]].sum()
        d += distance[tsp_individ[-1], tsp_individ[0]]
        return d

    # ---------------- Standard TTP (profit/time/weight) ----------------
    def __computeIndividStandardTTP(self, individ, v_min=0.1, v_max=1.0, Wmax=25936):
        tsp = individ["tsp"]
        kp  = individ["kp"]

        Wcur = 0.0
        Tcur = 0.0
        Pcur = 0.0

        for i in range(self.GENOME_LENGTH - 1):
            city = tsp[i]
            take = kp[city]

            if take:
                Pcur += self.item_profit[city]
                Wcur += self.item_weight[city]

            v = self.computeSpeedTTP(Wcur, v_max, v_min, Wmax)
            Tcur += self.distance[city, tsp[i + 1]] / v

        v = self.computeSpeedTTP(Wcur, v_max, v_min, Wmax)
        Tcur += self.distance[tsp[-1], tsp[0]] / v

        Dcur = self.computeIndividDistance(tsp)
        return Pcur, Tcur, Wcur, Dcur

    def metricsTTPStandard(self, genomics, **kw):
        # accept **kw so passing alpha/W doesn't crash
        v_min = kw.get("v_min", 0.1)
        v_max = kw.get("v_max", 1.0)
        Wmax  = kw.get("Wmax", kw.get("W", 25936))

        profits  = np.zeros(self.POPULATION_SIZE, dtype=np.float64)
        times    = np.zeros(self.POPULATION_SIZE, dtype=np.float64)
        weights  = np.zeros(self.POPULATION_SIZE, dtype=np.float64)
        dist_tsp = np.zeros(self.POPULATION_SIZE, dtype=np.float64)

        for idx, individ in enumerate(genomics.population()):
            p, t, w, d = self.__computeIndividStandardTTP(
                individ, v_min=v_min, v_max=v_max, Wmax=Wmax
            )
            profits[idx]  = p
            times[idx]    = t
            weights[idx]  = w
            dist_tsp[idx] = d

        metric_values = {
            "profit": profits,
            "time": times,
            "weight": weights,
            "distance": dist_tsp
        }
        return metric_values

    # ---------------- TTP Linear ----------------
    def __computeIndividLiniarTTP(self, individ, *args, v_min=0.1, v_max=1, W=2000, alpha=0.01):
        Wcur = 0.0
        Tcur = 0.0
        Pcur = 0.0

        tsp_individ = individ["tsp"]
        kp_individ  = individ["kp"]

        distance, item_profit, item_weight = args

        for i in range(self.GENOME_LENGTH-1):
            city = tsp_individ[i]
            take = kp_individ[city]

            profit = item_profit[city]*take
            weight = item_weight[city]*take

            Pcur += max(0.0, profit - alpha*Tcur)
            Wcur += weight

            v = v_max - v_min * (Wcur / W)
            v = max(v_min, v)

            Tcur += distance[city, tsp_individ[i+1]] / v

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

        number_city = self.computeNumberCities(genomics.chromosomes("tsp"))

        metric_values = {
            "profits"    : profits,
            "times"      : times,
            "weights"    : weights,
            "number_city": number_city
        }
        return metric_values

    # ---------------- TTP Adaptive Linear ----------------
    def __computeIndividAdaLiniarTTP(self, individ, *args, v_min=0.1, v_max=1, W=2000, alpha=0.01):
        Wcur = 0.0
        Tcur = 0.0
        Pcur = 0.0

        tsp_individ = individ["tsp"]
        kp_individ  = individ["kp"]

        distance, item_profit, item_weight = args

        for i in range(self.GENOME_LENGTH-1):
            city = tsp_individ[i]
            take = kp_individ[city]

            profit = item_profit[city]*take
            weight = item_weight[city]*take

            Pcur += max(0.0, profit**2/(weight+1e-7) - alpha*Tcur)
            Wcur += weight

            v = v_max - v_min * ((Wcur / W) - 1.)
            v = max(v_min, v)

            Tcur += distance[city, tsp_individ[i+1]] / v

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

        number_city = self.computeNumberCities(genomics.chromosomes("tsp"))

        number_obj  = self.computeNbrObjKP(genomics.chromosomes("kp"))
        number_obj  = number_obj*kw.get("W")/((weights + 1e-7)*self.GENOME_LENGTH)

        mask = (kw.get("W") <= weights)
        number_obj[mask] = 1.

        metric_values = {
            "profits"    : profits,
            "times"      : times,
            "weights"    : weights,
            "number_city": number_city,
            "number_obj" : number_obj
        }
        return metric_values

    # ---------------- TTP Exponential ----------------
    def __computeIndividExpTTP(self, individ, *args, v_min=0.1, v_max=1, W=2000, lam=0.01):
        Wcur = 0.0
        Tcur = 0.0
        Pcur = 0.0

        tsp_individ = individ["tsp"]
        kp_individ  = individ["kp"]

        distance, item_profit, item_weight = args

        for i in range(self.GENOME_LENGTH-1):
            city = tsp_individ[i]
            take = kp_individ[city]

            profit = item_profit[city]*take
            weight = item_weight[city]*take

            Pcur += profit * np.exp(-lam * Tcur)
            Wcur += weight

            v = v_max - (v_max - v_min) * (Wcur / W)
            Tcur += distance[city, tsp_individ[i+1]] / v

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

        metric_values = {
            "profits"  : profits,
            "weights"  : weights,
            "times"    : times
        }
        return metric_values

    # ---------------- Score extraction (patched to include time) ----------------
    def getScoreTTP(self, genomics, fitness_values):
        arg_best = self.getArgBest(fitness_values)
        individ  = genomics[arg_best]
        best_fitness = fitness_values[arg_best]
        genomics.setBest(individ)

        tsp_individ = individ["tsp"]
        kp_individ  = individ["kp"]

        distance = self.computeIndividDistance(tsp_individ)
        profit   = self.computeIndividProfitKP(kp_individ)
        weight   = self.computeIndividWeightKP(kp_individ)

        time_val = None
        if isinstance(self.metrics_cache, dict) and ("time" in self.metrics_cache):
            try:
                time_val = float(self.metrics_cache["time"][arg_best])
            except Exception:
                time_val = None

        return {
            "score": profit,
            "distance": distance,
            "weight": weight,
            "time": time_val,
            "profit" : profit,
            "best_fitness": best_fitness
        }

def normalization(x):
    x_min = x.min()
    x_max = x.max()
    denom = x_max - x_min
    if (denom == 0):
        x_ret = np.ones_like(x, dtype=np.float32)
    else:
        x_ret = (x_max-x)/(x_max-x_min)
    return x_ret
