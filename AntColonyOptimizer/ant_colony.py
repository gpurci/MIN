#!/usr/bin/python

import numpy as np
from extension.utils.normalization import *

class AntColonyOptimization(object):
    def __init__(self, distance, genome_length,
                        alpha=1.0, beta=5.0,
                        rho=0.5, q=1.0):

        self.distance = distance
        self.GENOME_LENGTH = genome_length

        # parameters
        self.alpha = alpha  # importance of pheromone
        self.beta  = beta   # importance of heuristic
        self.rho   = rho    # evaporation rate
        self.q     = q      # pheromone deposit factor

        # pheromone and heuristic
        self.eta = (1. / (self.distance + 1e-7))  # heuristic: 1 / distance
        self.tau = np.ones((self.GENOME_LENGTH, self.GENOME_LENGTH))
        # monitor the best start city
        self.map_start_city = np.ones(self.GENOME_LENGTH, dtype=np.float32)
        self.start_city     = np.random.randint(low=0, high=self.GENOME_LENGTH, size=None, dtype=int)
    # ======================================================================

    # ----------------------------------------------------------------------
    def fill_chromosome(self, route, start_gene, allowed_city):
        # make route
        for i in range(start_gene, self.GENOME_LENGTH-1):
            current = route[i]
            # allowed cities
            allowed = np.argwhere(allowed_city).reshape(-1)
            # transition probabilities
            tau_vals = self.tau[current, allowed] ** self.alpha
            eta_vals = self.eta[current, allowed] ** self.beta
            # calculeaza probabilitatea selectarii urmatorului oras
            probs  = tau_vals * eta_vals
            total_probs = probs.sum()
            if (total_probs != 0):
                probs /= total_probs
            else:
                probs  = eta_vals / eta_vals.sum()
            # selecteaza urmatorul oras
            next_city  = np.random.choice(allowed, p=probs.reshape(-1))
            route[i+1] = next_city
            allowed_city[next_city] = False
        return route
    # ======================================================================

    # ----------------------------------------------------------------------
    def _construct_solution(self, path=None, start_city=0):
        """ Construct a TSP tour for one ant """
        self.start_city = (self.start_city + 1) % self.GENOME_LENGTH
        rand_start_city = self.start_city
        # init route
        route    = np.zeros(self.GENOME_LENGTH, dtype=np.int32)
        route[0] = rand_start_city
        # allowed city
        allowed_city = np.ones(self.GENOME_LENGTH, dtype=bool)
        allowed_city[rand_start_city] = False
        route = self.fill_chromosome(route, 0, allowed_city)
        # 
        size_shift = np.argwhere(route == start_city).reshape(-1)
        route = np.roll(route, -size_shift)
        return route
    # ======================================================================

    # ----------------------------------------------------------------------
    def _penalization(self, route, penalization):
        start_city = np.arange(self.GENOME_LENGTH, dtype=np.int32)
        next_city  = np.roll(start_city, -1)
        self.tau[start_city, next_city] *= penalization
    # ======================================================================

    # ----------------------------------------------------------------------
    def _penalize_one_city(self, start_city, next_city, penalization=0.1):
        self.tau[start_city, next_city] *= penalization
    # ======================================================================

    # ----------------------------------------------------------------------
    def _deposit(self, best_path, best_cost, q, rho):
        # Evaporation
        self.tau *= (1 - rho)
        # Deposit pheromone on edges of best ant
        deposit   = q / best_cost
        # deposit 
        start_city = np.arange(self.GENOME_LENGTH, dtype=np.int32)
        start_path = best_path[start_city]
        next_path  = np.roll(start_path, -1)
        self.tau[start_path, next_path ] += deposit
        self.tau[next_path,  start_path] += deposit
    # ======================================================================

    # ----------------------------------------------------------------------
    def __call__(self, start_city, generations, size_ants):
        # init potential population
        all_paths = np.zeros((size_ants, self.GENOME_LENGTH), dtype=np.int32)
        # find best route
        for epoch in range(generations):
            # find the best 'size_ants' routes
            yield epoch, np.apply_along_axis(self._construct_solution,
                                        axis=1,
                                        arr=all_paths,
                                        start_city=start_city)
    # ======================================================================
