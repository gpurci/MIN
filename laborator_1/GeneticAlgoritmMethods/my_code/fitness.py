#!/usr/bin/python

import numpy as np 
from root_GA import *

class Fitness(RootGA):
    """
    Clasa 'Fitness', ofera doar metode pentru a calcula functia fitness din populatie.
    Functia 'fitness' are 1 parametru, numarul populatiei.
    Metoda '__config_fn', selecteaza functia de initializare.
    Metoda '__call__', aplica functia de initializare ce a fost selectata in '__config_fn'
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """

    def __init__(self, config, metrics):
        super().__init__()
        self.__config = None
        self.metrics = metrics
        self.setConfig(config)

    def __call__(self, population):
        return self.fn(population)

    def __config_fn(self):
        self.fn = self.fitnessAbstract
        if self.__config is not None:
            if self.__config == "TSP_f1score":
                self.fn = self.fitnessF1scoreTSP
            elif self.__config == "TTP_linear":
                self.fn = self.fitness_ttp_linear
            elif self.__config == "TTP_exp":
                self.fn = self.fitness_ttp_exp

    def help(self):
        info = """Fitness:
        metode de config: 'TSP_f1score', 'TTP_linear', 'TTP_exp'
        """
        return info

    def setConfig(self, config):
        self.__config = config
        self.__config_fn()

    def fitnessAbstract(self, size):
        raise NameError("Lipseste configuratia pentru functia de 'Fitness': config '{}'".format(self.__config))

    # TS problem------------------------------
    def fitnessF1scoreTSP(self, population):
        """ Returneaza o valoare normalizata, formula 2*weights*profits/(weights+profits)
        unde: valoarea distantei este invers normalizata, iar valoarea numarului de orase direct normalizata
        population - populatia, vector de indivizi
        """
        metrics_values = self.metrics(population)
        # calculeaza distanta
        distances = metrics_values["distances"]
        # calculeaza numarul de orase unice
        number_city = metrics_values["number_city"]
        # normalizeaza intervalul 0...1
        number_city = self.__cityNormTSP(number_city)
        distances = self.__distanceNormTSP(distances)
        fitness_values = 2 * distances * number_city / (distances + number_city + 1e-7)
        #print("fitness {}".format(fitness_values.shape))
        return fitness_values

    def __distanceNormTSP(self, distances):
        mask_not_zero = (distances != 0)
        valid_distances = distances[mask_not_zero]
        if valid_distances.shape[0] > 0:
            self.__min_distance = valid_distances.min()
        else:
            self.__min_distance = 0.1
            distances[:] = 0.1
        return (2 * self.__min_distance) / (distances + self.__min_distance)

    def __cityNormTSP(self, number_city):
        mask_cities = (number_city == self.GENOME_LENGTH)
        return mask_cities.astype(np.float32)

    # TP problem=================================

    # TTP fitness (linear)
    def fitness_ttp_linear(self, population):
        fitness_values = np.zeros(population.shape[0], dtype=float)

        for idx, individ in enumerate(population):

            current_weight = 0.0
            total_profit = 0.0
            total_time = 0.0

            for i in range(len(individ) - 1):
                city = individ[i]

                # take items in this city
                for (city_k, weight_k, profit_k) in self.items:
                    if city_k == city:
                        p = profit_k - self.alpha * total_time
                        if p < 0.0:
                            p = 0.0
                        total_profit += p
                        current_weight += weight_k

                # speed = f(weight)
                v = self.v_max - (self.v_max - self.v_min) * (current_weight / self.W)

                total_time += self.distance[individ[i], individ[i + 1]] / v

            v = self.v_max - (self.v_max - self.v_min) * (current_weight / self.W)
            total_time += self.distance[individ[-1], individ[0]] / v

            fitness_values[idx] = total_profit - self.R * total_time

        return fitness_values

    # =====================================================
    # TTP fitness (exponential)
    def fitness_ttp_exp(self, population):
        fitness_values = np.zeros(population.shape[0], dtype=float)

        for idx, individ in enumerate(population):

            current_weight = 0.0
            total_profit = 0.0
            total_time = 0.0

            for i in range(len(individ) - 1):
                city = individ[i]

                for (city_k, weight_k, profit_k) in self.items:
                    if city_k == city:
                        p = profit_k * np.exp(-self.lam * total_time)
                        total_profit += p
                        current_weight += weight_k

                v = self.v_max - (self.v_max - self.v_min) * (current_weight / self.W)
                total_time += self.distance[individ[i], individ[i + 1]] / v

            v = self.v_max - (self.v_max - self.v_min) * (current_weight / self.W)
            total_time += self.distance[individ[-1], individ[0]] / v

            fitness_values[idx] = total_profit - self.R * total_time

        return fitness_values

    def setTTPParams(self, *, distance, items, v_min, v_max, W, R, alpha=0.1, lam = 0.001):
        self.distance = distance
        self.items    = items
        self.v_min    = v_min
        self.v_max    = v_max
        self.W        = W
        self.R        = R
        self.alpha    = alpha
        self.lam      = lam
