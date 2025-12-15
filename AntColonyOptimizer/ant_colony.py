#!/usr/bin/python

import numpy as np
from extension.utils.normalization import *

class AntColonyOptimization(object):
    def __init__(self, dataset_man, 
                        alpha=1.0, beta=5.0,
                        rho=0.5, q=1.0):

        self.dataset_man   = dataset_man
        self.min_distance  = dataset_man.neighborsDistance(1).reshape(-1)
        self.GENOME_LENGTH = dataset_man.GENOME_LENGTH

        # parameters
        self.alpha = alpha      # importance of pheromone
        self.beta = beta        # importance of heuristic
        self.rho = rho          # evaporation rate
        self.q = q              # pheromone deposit factor

        # pheromone and heuristic
        self.eta = (1. / (self.dataset_man.distance + 1e-7))  # heuristic: 1 / distance
        self.tau = np.ones((self.GENOME_LENGTH, self.GENOME_LENGTH))
        # best-so-far memory
        self.best_path = None
        self.best_cost = np.inf
        self.map_start_city = np.ones(self.GENOME_LENGTH, dtype=np.float32)
        self.start_city = np.random.randint(low=0, high=self.GENOME_LENGTH, size=None, dtype=int)
    # ======================================================================

    # ----------------------------------------------------------------------
    def _construct_solution(self, start_city):
        """Construct a TSP tour for one ant."""
        self.start_city = (self.start_city + 1) % self.GENOME_LENGTH
        rand_start_city = self.start_city
        route    = np.zeros(self.GENOME_LENGTH, dtype=np.int32)
        route[0] = rand_start_city
        x_range  = np.arange(self.GENOME_LENGTH)
        # allowed city
        allowed_city = np.ones(self.GENOME_LENGTH, dtype=bool)
        allowed_city[rand_start_city] = False

        for i in range(self.GENOME_LENGTH-1):
            current = route[i]
            # allowed cities
            allowed = np.argwhere(allowed_city).reshape(-1)
            # transition probabilities
            tau_vals = self.tau[current, allowed] ** self.alpha
            eta_vals = self.eta[current, allowed] ** self.beta
            # calculeaza probabilitatea selectarii urmatorului oras
            probs  = tau_vals * eta_vals
            probs /= probs.sum()
            # selecteaza urmatorul oras
            next_city  = np.random.choice(allowed, p=probs.reshape(-1))
            route[i+1] = next_city
            allowed_city[next_city] = False
        # 
        size_shift = np.argwhere(route == start_city).reshape(-1)
        route = np.roll(route, -size_shift)
        return route
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
    def _propagate_corection_fn(self, route):
        city_distances = self.dataset_man.computeIndividDistanceFromCities(route)
        repay          = self.min_distance[route] / city_distances
        # corecteaza fata de cea mai mare distanta
        arg_min = np.argmin(repay)
        back_propagate   = np.linspace(start=1, stop=repay[arg_min], num=arg_min+1)
        #back_propagate   = np.sqrt(back_propagate)
        repay[:arg_min] *= back_propagate[:-1]
        return repay
    # ======================================================================

    # ----------------------------------------------------------------------
    def _corection_fn(self, route):
        city_distances = self.dataset_man.computeIndividDistanceFromCities(route)
        repay          = self.min_distance[route] / city_distances
        return repay
    # ======================================================================

    # ----------------------------------------------------------------------
    def update_tau_apply_corection(self, route, corection):
        for i in range(self.GENOME_LENGTH-1):
            a = route[i]
            b = route[i+1]
            self.tau[a, b] *= corection[i]
            self.tau[b, a] *= corection[i]
        else:
            a = route[-1]
            b = route[0]
            self.tau[a, b] *= corection[i]
            self.tau[b, a] *= corection[i]
    # ======================================================================

    # ----------------------------------------------------------------------
    def _deposit(self, best_path, best_cost, q):
        # Deposit pheromone on edges of best ant
        deposit   = q / best_cost
        # deposit 
        self.tau[best_path[:-1], best_path[ 1:]] += deposit
        self.tau[best_path[ 1:], best_path[:-1]] += deposit
        self.tau[best_path[-1], best_path[ 0]] += deposit
        self.tau[best_path[ 0], best_path[-1]] += deposit
    # ======================================================================

    # ----------------------------------------------------------------------
    def _update_pheromones(self, best_path, best_cost):
        # Evaporation
        self.tau *= (1 - self.rho)

        # Deposit pheromone on edges of best ant
        self._deposit(best_path, best_cost, self.q)
        corection = self._propagate_corection_fn(best_path)
        self.update_tau_apply_corection(best_path, corection)

        # Update global best
        if (best_cost < self.best_cost):
            self.best_cost = best_cost
            self.best_path = best_path.copy()
            self.map_start_city[self.start_city] += 1
    # ======================================================================

    # ----------------------------------------------------------------------
    def _updateByElite(self, population):
        self.tau = np.ones((self.GENOME_LENGTH, self.GENOME_LENGTH))
        all_costs = []
        for individ in population:
            cost = self.dataset_man.computeIndividDistance(individ)
            all_costs.append(cost)
        if (len(population) > 0): # update 
            all_costs = np.array(all_costs)
            argsort_cost = np.argsort(all_costs).reshape(-1)
            argmin_cost  = argsort_cost[0]
            best_cost = all_costs[argmin_cost]
            best_path = population[argmin_cost]
            # deposit feromones
            for idx in argsort_cost:
                cost = all_costs[idx]
                path = population[idx]
                q = 0.1
                self._deposit(path, cost, q)
            # corection
            for idx in argsort_cost:
                individ   = population[idx]
                corection = self._propagate_corection_fn(individ)
                #corection = np.sqrt(corection)
                self.update_tau_apply_corection(individ, corection)
            #self.tau = np.cbrt(self.tau)
    # ======================================================================

    # ----------------------------------------------------------------------
    def updateByElite(self, population):
        self.tau = np.ones((self.GENOME_LENGTH, self.GENOME_LENGTH))
        all_costs = []
        for individ in population:
            cost = self.dataset_man.computeIndividDistance(individ)
            all_costs.append(cost)
        if (len(population) > 0): # update 
            all_costs = np.array(all_costs)
            argsort_cost = np.argsort(all_costs).reshape(-1)
            for idx in argsort_cost:
                cost = all_costs[idx]
                path = population[idx]
                self._update_pheromones(path, cost)
    # ======================================================================

    def __call__(self, routes, start_city, generations, size_ants, monitor_size):
        self._updateByElite(routes)
        p_select  = 1 - (self.map_start_city / self.map_start_city.sum())
        p_select /= p_select.sum()
        #self.start_city = np.random.choice(self.GENOME_LENGTH, p=p_select)
        print("Start city: {}, prob: {}, freq: {}".format(self.start_city, p_select[self.start_city], self.map_start_city[self.start_city]))
        # best-so-far memory
        self.best_path = None
        self.best_cost = np.inf
        self._evolution_monitor = np.zeros(monitor_size, dtype=np.float32)
        for epoch in range(generations):
            all_paths = []
            all_costs = []

            for _ in range(size_ants):
                path = self._construct_solution(start_city)
                cost = self.dataset_man.computeIndividDistance(path)

                all_paths.append(path)
                all_costs.append(cost)
            best_path, best_cost = self._find_best_path(all_paths, all_costs)
            self._update_pheromones(best_path, best_cost)
            print("generatia: {}, best cost: {}".format(epoch, best_cost))
            self._evolution_monitor[:-1] = self._evolution_monitor[1:]
            self._evolution_monitor[-1]  = best_cost
            if (np.allclose(self._evolution_monitor, best_cost, rtol=1e-03, atol=1e-07)):
                break
        return self.best_path
