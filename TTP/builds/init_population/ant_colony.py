#!/usr/bin/python

import numpy as np
from sys_function import sys_remove_modules

from extension.utils.normalization import *

sys_remove_modules("AntColonyOptimizer.ant_colony")
from AntColonyOptimizer.ant_colony import *

class AntColonyInitTTP(AntColonyOptimization):
    def __init__(self, metrics, alpha=1.0, beta=5.0, rho=0.5, q=1.0):
        super().__init__(metrics.distance, metrics.GENOME_LENGTH, 
                            alpha=alpha, beta=beta, rho=rho, q=q)

        self.metrics      = metrics
        self.min_distance = metrics.neighborsDistance(1).reshape(-1)

        # best-so-far memory
        self.best_path = None
        self.best_cost = np.inf
    # ======================================================================

    # ----------------------------------------------------------------------
    def _find_best_path(self, all_paths, all_costs):
        # Best ant (iteration-best)
        idx = np.argmin(all_costs)
        best_path = all_paths[idx]
        best_cost = all_costs[idx]
        return best_path, best_cost
    # ======================================================================

    # ----------------------------------------------------------------------
    def _perform_penalization(self, route):
        city_distances = self.metrics.computeIndividDistanceFromCities(route)
        repay          = self.min_distance[route] / city_distances
        # corecteaza fata de cea mai mare distanta
        argsort = np.argsort(repay)
        
        for arg in argsort[-3:]:
            back_propagate = np.linspace(start=1, stop=repay[arg], num=arg)
            back_propagate = np.cbrt(back_propagate)
            repay[:arg] *= back_propagate
        repay[argsort[-3:]] = 0.
        return repay
    # ======================================================================

    # ----------------------------------------------------------------------
    def _update_pheromones(self, path, cost):
        # Deposit pheromone on edges of best ant
        self._deposit(path, cost, self.q, self.rho)
        penalization = self._perform_penalization(path)
        self._penalization(path, penalization)

        # Update global best
        if (cost < self.best_cost):
            self.best_cost = cost
            self.best_path = path.copy()
            self.map_start_city[self.start_city] += 1
    # ======================================================================

    # ----------------------------------------------------------------------
    def _updateByElite(self, population):
        self.tau = np.ones((self.GENOME_LENGTH, self.GENOME_LENGTH))
        all_costs = self.metrics.computeDistances(population)
        if (len(population) > 0): # update
            argsort_cost = np.argsort(all_costs).reshape(-1)
            # deposit feromones
            for idx in argsort_cost:
                cost = all_costs[idx]
                path = population[idx]
                q = 0.1
                self._deposit(path, cost, q)
            # penalization
            for idx in argsort_cost:
                individ      = population[idx]
                penalization = self._perform_penalization(individ)
                self._penalization(individ, penalization)
    # ======================================================================

    # ----------------------------------------------------------------------
    def updateByElite(self, population, q=1.0, rho=0.9, size_best=3):
        self.tau = np.ones((self.GENOME_LENGTH, self.GENOME_LENGTH))
        similar_args_flag = self.similarIndivids(population)
        mask = np.invert(similar_args_flag)
        print("mask {}", mask, )
        population = population[mask]
        all_costs  = self.metrics.computeDistances(population)
        if (len(population) > 0): # update 
            argsort_cost = np.argsort(all_costs).reshape(-1)
            for idx in argsort_cost[-size_best:]:
                cost = all_costs[idx]
                path = population[idx]
                self._deposit(path, cost, q, rho)
    # ======================================================================

    def findSimilarIndivids(self, population, individ, tolerance):
        """
        Cauta indivizi din intreaga populatie ce are codul genetic identic cu un individ,
        population - lista de indivizi
        individ    - vector compus din codul genetic
        tolerance  - cate gene pot fi diferite
        """
        tmp = (population==individ).sum(axis=1)
        return np.argwhere(tmp>=tolerance)

    def similarIndivids(self, population):
        """
        Returneaza un vector de flaguri pentru fiecare individ din populatie daca este gasit codul genetic si la alti indivizi
        population - lista de indivizi
        """
        # initializare vector de flaguri pentru fiecare individ
        similar_args_flag = np.zeros(population.shape[0], dtype=bool)
        # setare toleranta, numarul total de gene
        tolerance = population.shape[1]
        # 
        for i in range(population.shape[0]):
            if (similar_args_flag[i]):
                pass
            else:
                individ = population[i]
                similar_args = self.findSimilarIndivids(population, individ, tolerance)
                #print("similar_args", similar_args)
                similar_args_flag[similar_args] = True
                similar_args_flag[i] = False # scoate flagul de pe individul care este copiat
        return similar_args_flag

    # ----------------------------------------------------------------------
    def __call__(self, start_city, generations, size_ants, monitor_size):
        # best-so-far memory
        self.best_path = None
        self.best_cost = np.inf
        self._evolution_monitor = np.zeros(monitor_size, dtype=np.float32)
        for epoch, all_paths in super().__call__(start_city, generations, size_ants):
            all_costs = self.metrics.computeDistances(all_paths)
            best_path, best_cost = self._find_best_path(all_paths, all_costs)
            self._update_pheromones(best_path, best_cost)
            print("generatia: {}, best cost: {}".format(epoch, best_cost))
            self._evolution_monitor[:-1] = self._evolution_monitor[1:]
            self._evolution_monitor[-1]  = best_cost
            if (np.allclose(self._evolution_monitor, best_cost, rtol=1e-03, atol=1e-07)):
                break
        return self.best_path
    # ======================================================================
