#!/usr/bin/python

import numpy as np

class AntColonyOptimization(object):
    def __init__(self, dataset_man, alpha=1.0, beta=5.0,
                 rho=0.5, q=1.0):

        self.dataset_man   = dataset_man
        self.GENOME_LENGTH = dataset_man.GENOME_LENGTH

        # parameters
        self.alpha = alpha      # importance of pheromone
        self.beta = beta        # importance of heuristic
        self.rho = rho          # evaporation rate
        self.q = q              # pheromone deposit factor

        # pheromone and heuristic
        self.eta = 1. / (self.dataset_man.distance + 1e-7)  # heuristic: 1 / distance

        # best-so-far memory
        self.best_path = None
        self.best_cost = np.inf

    # ----------------------------------------------------------------------

    def _construct_solution(self, start_city):
        """Construct a TSP tour for one ant."""
        route    = np.zeros(self.GENOME_LENGTH, dtype=np.int32)
        route[0] = start_city
        x_range  = np.arange(self.GENOME_LENGTH)
        # allowed city
        allowed_city = np.ones(self.GENOME_LENGTH, dtype=bool)
        allowed_city[start_city] = False

        for i in range(self.GENOME_LENGTH-1):
            current = route[i]
            # allowed cities
            allowed = np.argwhere(allowed_city)
            # transition probabilities
            tau_vals = self.tau[current, allowed] ** self.alpha
            eta_vals = self.eta[current, allowed] ** self.beta
            # calculeaza probabilitatea selectarii urmatorului oras
            probs  = tau_vals * eta_vals
            probs /= probs.sum()
            # selecteaza urmatorul oras
            next_city  = np.random.choice(allowed, p=probs)
            route[i+1] = next_city
            allowed_city[next_city] = False

        return route

    # ----------------------------------------------------------------------

    def _update_pheromones(self, all_paths, all_costs):
        # Evaporation
        self.tau *= (1 - self.rho)

        # Best ant (iteration-best)
        idx = np.argmin(all_costs)
        best_path = all_paths[idx]
        best_cost = all_costs[idx]

        # Deposit pheromone on edges of best ant
        deposit = self.q / best_cost
        for i in range(self.GENOME_LENGTH-1):
            a = best_path[i]
            b = best_path[i+1]
            self.tau[a, b] += deposit
            self.tau[b, a] += deposit
        else:
            a = best_path[-1]
            b = best_path[0]
            self.tau[a, b] += deposit
            self.tau[b, a] += deposit

        # Update global best
        if (best_cost < self.best_cost):
            self.best_cost = best_cost
            self.best_path = best_path.copy()

    # ----------------------------------------------------------------------

    def __call__(self, route, city_distances, start_city, generations, size_ants):
        self.tau = np.ones((self.GENOME_LENGTH, self.GENOME_LENGTH))

        for i in range(self.GENOME_LENGTH-1):
            a = route[i]
            b = route[i+1]
            self.tau[a, b] *= city_distances[i]
            self.tau[b, a] *= city_distances[i]
        else:
            a = route[-1]
            b = route[0]
            self.tau[a, b] *= city_distances[i]
            self.tau[b, a] *= city_distances[i]

        for _ in range(generations):
            all_paths = []
            all_costs = []

            for _ in range(size_ants):
                path = self._construct_solution(start_city)
                cost = self.dataset_man.computeIndividDistance(path)

                all_paths.append(path)
                all_costs.append(cost)

            self._update_pheromones(all_paths, all_costs)
        del self.tau
        return self.best_path
