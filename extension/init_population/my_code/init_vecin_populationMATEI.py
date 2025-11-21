#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import *

class InitVecinPopulation(RootGA):
    """
    Extension: Greedy neighbor-based TTP initial population (TTP_vecin),
    self-contained version (NO dependency on Metrics).
    """

    def __init__(self, method="TTP_vecin", dataset=None, **configs):
        super().__init__()
        self.__configs = configs
        self.__method = method

        if dataset is None:
            raise ValueError("InitVecinPopulation requires dataset=<TTP dataset dict>")

        self.dataset = dataset
        self.distance    = dataset["distance"]
        self.item_profit = dataset["item_profit"]
        self.item_weight = dataset["item_weight"]
        self.__fn = self.initPopulationTTP

    def __str__(self):
        return f"InitVecinPopulation(method={self.__method}, configs={self.__configs})"

    def help(self):
        print("""InitVecinPopulation:
    metoda: 'TTP_vecin'; config:
        lambda_time=0.1, vmax=1.0, vmin=0.1, Wmax=25936, seed\n""")

    def __call__(self, size, genoms=None):
        self.__fn(size, genoms=genoms, **self.__configs)

    def initPopulationTTP(self, size, genoms=None,
                          lambda_time=0.1, vmax=1.0, vmin=0.1, Wmax=25936, seed=None):

        if seed is not None:
            np.random.seed(seed)

        seen = set()
        count = 0

        while count < size:
            start_city = 0

            route, kp = self._constructGreedyRoute(
                start_city, lambda_time, vmax, vmin, Wmax
            )

            # two-opt without metrics
            route = self._twoOpt(route)

            tup = tuple(route)
            if tup in seen:
                continue
            seen.add(tup)

            genoms.add(tsp=route, kp=kp)
            count += 1

        genoms.saveInit()
        print("Greedy TTP population =", genoms.shape)

    def computeSpeedTTP(self, Wcur, v_max, v_min, Wmax):
        return v_max - (v_max - v_min) * (Wcur / float(Wmax))

    def _route_distance(self, route):
        d = self.distance[route[:-1], route[1:]].sum()
        d += self.distance[route[-1], route[0]]
        return d

    def _twoOpt(self, route):
        best = route.copy()
        best_d = self._route_distance(best)
        n = len(best)

        for i in range(1, n - 2):
            for k in range(i + 1, n - 1):
                new_r = best.copy()
                new_r[i:k + 1] = new_r[i:k + 1][::-1]

                d = self._route_distance(new_r)
                if d < best_d:
                    return new_r

        return best

    def _constructGreedyRoute(self, start, lambda_time, vmax, vmin, Wmax):
        n = self.GENOME_LENGTH
        visited = np.zeros(n, dtype=bool)
        visited[start] = True

        path = [start]
        kp = np.zeros(n, dtype=np.int32)
        cur = start
        Wcur = 0.0

        for _ in range(n - 1):
            cand = np.where(~visited)[0]

            v_cur = self.computeSpeedTTP(Wcur, vmax, vmin, Wmax)

            dist = self.distance[cur, cand]
            time = dist / v_cur

            profit_raw = self.item_profit[cand]
            profit_adj = np.maximum(0.0, profit_raw - lambda_time * time)

            feasible = (Wcur + self.item_weight[cand]) <= Wmax
            score = profit_adj * feasible

            if np.all(score <= 0):
                j = cand[np.argmin(dist)]
            else:
                order = np.argsort(score)
                top = cand[order[-min(5, len(order)):]]
                j = np.random.choice(top)

            path.append(j)
            visited[j] = True

            gain = self.item_profit[j] - lambda_time * self.distance[cur, j] / v_cur
            if gain > 0 and (Wcur + self.item_weight[j] <= Wmax):
                Wcur += self.item_weight[j]
                kp[j] = 1

            cur = j

        return np.array(path, dtype=np.int32), kp
