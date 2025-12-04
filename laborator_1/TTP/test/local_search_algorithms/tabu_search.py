#!/usr/bin/python
import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import *


class TabuSearch(RootGA):

    def __init__(self, method, dataset, **configs):
        super().__init__()
        self.dataset = dataset
        self.dist = dataset["distance"]

        self.method = method
        self.max_iter = configs.get("max_iter", 50)
        self.tabu_size = configs.get("tabu_size", 100)
        self.configs = configs

        self.fn = self._select_method(method)

    def _select_method(self, method):
        table = {
            "tabu_search": self.tabuSearch,
            "tabu_search_rand": self.tabuSearchRand,
            "tabu_search_distance": self.tabuSearchDistance,
            "tabu_2opt_fast": self.tabu_2opt_fast,
        }
        return table.get(method, self.tabuSearchAbstract)

    def __str__(self):
        return f"TabuSearch(method={self.method}, configs={self.configs})"

    def __call__(self, route):
        return self.fn(route)

    # ------------------ 2-opt delta + apply (INSIDE class) ------------------
    def delta_2opt(self, route, i, j):
        dist = self.dist
        n = len(route)

        a = route[i - 1]
        b = route[i]
        c = route[j]
        d = route[(j + 1) % n]

        old1 = dist[a, b]
        old2 = dist[c, d]
        new1 = dist[a, c]
        new2 = dist[b, d]

        return (new1 + new2) - (old1 + old2)

    def apply_2opt(self, route, i, j):
        new = route.copy()
        new[i:j+1] = new[i:j+1][::-1]
        return new

    # ------------------ FAST tabu-ish 2-opt ------------------
    def tabu_2opt_fast(self, route):
        route = route.copy()
        best_cost = self._cost(route)
        best = route.copy()

        n = len(route)
        tabu = []  # store recent (i, j) moves

        for _ in range(self.max_iter):
            improved = False

            # try a limited number of random 2-opt candidates
            for _ in range(50):
                i = np.random.randint(1, n - 3)
                j = np.random.randint(i + 1, n - 1)

                if (i, j) in tabu:
                    continue

                delta = self.delta_2opt(route, i, j)

                if delta < 0:
                    route = self.apply_2opt(route, i, j)
                    best_cost += delta
                    best = route.copy()

                    tabu.append((i, j))
                    if len(tabu) > self.tabu_size:
                        tabu.pop(0)

                    improved = True
                    break

            if not improved:
                break

        # Optional safety check (cheap, can comment out):
        # if len(np.unique(best)) != len(best):
        #     raise ValueError("TabuSearch produced invalid permutation!")

        return best

    # ------------------ Old methods kept intact ------------------
    def tabuSearchAbstract(self, *args, **kw):
        raise NameError(f"Lipseste metoda '{self.method}', config: '{self.configs}'")

    def tabuSearch(self, route):
        best_score = self.computeIndividDistance(route)
        best = route.copy()
        for i in range(self.GENOME_LENGTH - 1):
            for j in range(i + 1, self.GENOME_LENGTH):
                tmp = route.copy()
                tmp[i], tmp[j] = tmp[j], tmp[i]
                score = self.computeIndividDistance(tmp)
                if score < best_score:
                    best_score = score
                    best = tmp
        return best

    def tabuSearchRand(self, route):
        start = np.random.randint(0, self.GENOME_LENGTH // 2)
        stop  = np.random.randint(start + self.GENOME_LENGTH // 4, self.GENOME_LENGTH)

        best_score = self.computeIndividDistance(route)
        best = route.copy()

        for i in range(start, stop - 1):
            for j in range(i + 1, stop):
                tmp = route.copy()
                tmp[i], tmp[j] = tmp[j], tmp[i]
                score = self.computeIndividDistance(tmp)
                if score < best_score:
                    best_score = score
                    best = tmp
        return best

    def tabuSearchDistance(self, route):
        city_d = self.individCityDistance(route)
        mask = city_d > city_d.mean()
        args = np.where(mask)[0]

        best_score = city_d.sum()
        best = route.copy()

        for i in range(len(args) - 1):
            for j in range(i + 1, len(args)):
                tmp = route.copy()
                a = args[i]
                b = args[j]
                tmp[a], tmp[b] = tmp[b], tmp[a]
                score = self.computeIndividDistance(tmp)
                if score < best_score:
                    best_score = score
                    best = tmp

        return best

    # ------------------ Utils ------------------
    def computeIndividDistance(self, r):
        d = self.dist
        return d[r[:-1], r[1:]].sum() + d[r[-1], r[0]]

    def individCityDistance(self, r):
        d = self.dist
        seg = d[r[:-1], r[1:]]
        last = d[r[-1], r[0]]
        return np.concatenate((seg, [last]))

    def _cost(self, route):
        d = self.dist
        return d[route[:-1], route[1:]].sum() + d[route[-1], route[0]]
