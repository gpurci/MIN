#!/usr/bin/python

import numpy as np
from my_code.root_GA import *

class InitPopulation(RootGA):
    """
    Clasa 'InitPopulation', ofera doar metode pentru a initializa populatia.
    Functia 'initPopulation' are 1 parametru, numarul populatiei.
    Metoda '__config_fn', selecteaza functia de initializare.
    Metoda '__call__', aplica functia de initializare ce a fost selectata in '__config_fn'
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self, config, metrics):
        self.metrics = metrics
        self.setConfig(config)

    def __call__(self, size):
        return self.fn(size)

    def __config_fn(self):
        self.fn = self.initPopulationAbstract
        if (self.__config is not None):
            if   (self.__config == "test"):
                self.fn = self.testParentClass
        else:
            pas

    def setConfig(self, config):
        self.__config = config
        self.__config_fn()

    def initPopulationAbstract(self, size):
        raise NameError("Lipseste configuratia pentru functia de 'InitPopulation': config '{}'".format(self.__config))


    def initPopulationMatei(self,
                    size=2000, lambda_time=0.1,
                    vmax=1.0, vmin=0.1, Wmax=25936, seed=None):

        if seed is not None:
            np.random.seed(seed)

        # load TTP instance (coords, distance matrix, item profit/weight)
        dataset = self.metrics.getDataset()
        self.coords      = dataset["coords"]
        self.distance    = dataset["distance"]
        self.item_profit = dataset["item_profit"]
        self.item_weight = dataset["item_weight"]

        population = []
        seen = set() # set of tuples (route) to reject duplicates
        starts = np.random.randint(0, self.GENOME_LENGTH, size=size) # draw random start cities for each individual

        for s in starts:
            # boolean mask = which cities are already visited
            visited = np.zeros(self.GENOME_LENGTH, dtype=bool)
            visited[s] = True

            path = [s]     # current route under construction

            cur  = s       # current knapsack state (for speed function)
            Wcur = 0.0     # accumulated knapsack weight
            Ptot = 0.0     # accumulated knapsack profit (not needed here, but kept for clarity)
            # greedy constructive heuristic (TTP-flavored)
            # build a route by inserting one city at a time
            # choose: best (profit - lambda * travel_time)
            for _ in range(self.GENOME_LENGTH-1):
                cand = np.where(~visited)[0]           # all unvisited cities

                # current TTP speed (vmax -> vmin depending on weight)
                frac  = min(1.0, Wcur/Wmax)
                v_cur = vmax - frac*(vmax-vmin)
                if v_cur < 1e-9: v_cur = 1e-9

                # travel time to each candidate city from current city
                dist = self.distance[cur, cand]
                time = dist / v_cur
                take_possible = (Wcur + self.item_weight[cand]) <= Wmax # boolean: can we take the item in that candidate city?
                profit_gain = self.item_profit[cand] * take_possible # effective profit gain if item fits
                score = profit_gain - lambda_time * time # utility score = profit gained - time penalty

                # pick the best candidate j = argmax score
                order = np.argsort(score)
                top_k = min(5, len(order))  # take the top 5 best candidates
                choices = cand[order[-top_k:]]
                j = np.random.choice(choices)

                # extend solution
                path.append(j)
                visited[j] = True

                # take item only if it fits
                if (Wcur + self.item_weight[j]) <= Wmax:
                    Wcur += self.item_weight[j]
                    Ptot += self.item_profit[j]
                cur = j

            # close tour by returning to start
            path.append(path[0])

            # small local improvement (2-opt) -> improves total tour length in O(n^2)
            path_np = np.array(path, dtype=np.int32)
            path_np = self.__twoOpt(path_np)
            tup = tuple(path_np)           # convert to hashable
            if tup in seen:
                continue                   # skip duplicates

            seen.add(tup)
            # add final route to population
            population.append(path_np)
            if len(population) >= size:
                break

        return np.array(population, dtype=np.int32)

    def __twoOpt(self, route):
        """
        single-pass 2-opt: tries one arc reversal,
        returns immediately if first improvement found.
        """
        best = route.copy()
        best_dist = self.___getIndividDistance(best)
        n = len(route) - 1

        for i in range(1, n-2):
            for k in range(i+1, n-1):
                new_route = best.copy()
                new_route[i:k] = best[k-1:i-1:-1]

                d = self.___getIndividDistance(new_route)
                if d < best_dist:
                    return new_route     # improvement found, exit early

        return best                     # no improvement found
    
    def ___getIndividDistance(self, individ):
        """Calculul distantei rutelor"""
        #print("individ", individ)
        distances = self.distance[individ[:-1], individ[1:]]
        distance = distances.sum() + self.distance[individ[-1], individ[0]]
        return distance