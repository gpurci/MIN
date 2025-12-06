#!/usr/bin/python

import numpy as np
from extension.dataset_utils.dataset_base import *

class DatasetTTPMan(DatasetBase):
    """
    """
    def __init__(self, dataset):
        super().__init__(dataset, "DatasetTTPMan")
        # 
        self.__distance    = dataset["distance"]
        self.__item_profit = dataset["item_profit"]
        self.__item_weight = dataset["item_weight"]
        self.GENOME_LENGTH = dataset["GENOME_LENGTH"]

    def getTupleDataset(self):
        # unpack datassets
        distance    = self.dataset["distance"]
        item_profit = self.dataset["item_profit"]
        item_weight = self.dataset["item_weight"]
        return (distance, item_profit, item_weight)

    def neighborsDistance(self, window_size):
        genom_length  = self.GENOME_LENGTH
        x_range       = np.arange(genom_length, dtype=np.int32)
        ret_neighbors = np.sort(self.__distance[x_range], axis=-1)[:, 1:window_size+1]
        return ret_neighbors

    def argsNeighborsDistance(self, window_size):
        genom_length  = self.GENOME_LENGTH
        x_range       = np.arange(genom_length, dtype=np.int32)
        ret_neighbors = np.argsort(self.__distance[x_range], axis=-1)[:, 1:window_size+1]
        return ret_neighbors

    def unvisitedNeighborDistance(self, city, window_size, visited_city):
        """Calculul distantei pentru un individ"""
        args  = np.argsort(self.__distance[city])
        count = 0
        ret_args = []
        for pos_city in args:
            if (visited_city[pos_city] == False):
                ret_args.append(pos_city)
                count += 1
            if (count >= window_size):
                break
        return np.array(ret_args, dtype=np.int32)

    # individ  ---------------------
    def computeIndividDistance(self, individ):
        distances = self.__distance[individ[:-1], individ[1:]].sum()
        return distances + self.__distance[individ[-1], individ[0]] # intoarcerea in orasul de start

    def computeIndividDistanceFromCities(self, individ):
        city_distances = self.__distance[individ[:-1], individ[1:]]
        to_first_city  = self.__distance[individ[-1], individ[0]]
        return np.concatenate((city_distances, [to_first_city]))

    def individDistanceFromCityToStart(self, individ):
        tmp = np.zeros(self.GENOME_LENGTH, dtype=np.float32)
        distances = self.computeIndividDistanceFromCities(individ)
        distance  = 0
        for i in range(self.GENOME_LENGTH-1, -1, -1):
            distance += distances[i]
            tmp[i]    = distance
        return tmp

    def computeIndividProfit(self, kp_individ):
        return (self.__item_profit*kp_individ).sum()

    def computeIndividWeight(self, kp_individ):
        return (self.__item_weight*kp_individ).sum()

    def argIndividMaxWeight(self, kp_individ):
        return np.argmax(self.__item_weight*kp_individ)

    def argminIndividProportion(self, kp_individ):
        proportion = self.calculateProportions()
        mask = np.invert(kp_individ.astype(bool))
        proportion[mask] = np.inf
        return np.argmin(proportion)

    def argsortIndividProportions(self, kp_individ):
        proportion = self.calculateProportions()
        mask = np.invert(kp_individ.astype(bool))
        proportion[mask] = -np.inf
        return np.argsort(proportion), mask.sum()

    def calculateIndividEarning(self, tsp_individ, kp_individ):
        proportion = self.calculateProportions()
        mask = np.invert(kp_individ.astype(bool))
        proportion[mask] = -np.inf
        proportion = proportion[tsp_individ]
        distances2start = self.individDistanceFromCityToStart(tsp_individ)
        proportion = proportion / distances2start
        return proportion, mask.sum()

    def argsortIndividEarning(self, tsp_individ, kp_individ):
        proportion, start_arg = self.calculateIndividEarning(tsp_individ, kp_individ)
        return np.argsort(proportion), start_arg

    def calculateIndividProportions(self, kp_individ):
        weight = self.__item_weight*kp_individ
        profit = self.__item_profit*kp_individ
        proportion = profit / (weight + 1e-7)
        return proportion

    def calculateProportions(self):
        weight = self.__item_weight
        profit = self.__item_profit
        proportion = profit / (weight + 1e-7)
        return proportion

    def computeIndividNbrObj(self, kp_individ):
        return kp_individ.sum()

    def computeIndividNumberCities(self, individ):
        return np.unique(individ, return_index=False, return_inverse=False, return_counts=False, axis=None).shape[0]

    def computeIndividSpeeds(self, tsp_individ, kp_individ, v_min=0.1, v_max=1, W=2000):
        # init viteza
        speeds = np.zeros_like(tsp_individ, dtype=np.float32)
        Wcur   = 0
        # vizităm secvenţial
        for i in range(self.GENOME_LENGTH):
            # get city
            city = tsp_individ[i]
            # take or not take object
            take = kp_individ[city]
            # calculate weight
            weight = self.__item_weight[city]*take
            Wcur  += weight
            # calculeaza viteza de tranzitie
            speed     = v_max - (v_max - v_min) * (Wcur / W)
            speeds[i] = max(v_min, speed)
        return speeds

    def computeIndividScore(self, tsp_individ, kp_individ, v_min=0.1, v_max=1, W=2000, R=1):
        # compute profit and weight
        profit = self.computeIndividProfit(kp_individ)
        # calculeaza viteza pentru cel mai bun individ
        speeds = self.computeIndividSpeeds(tsp_individ, kp_individ, v_min=v_min, v_max=v_max, W=W)
        # calculeaza distanta dintre orase pentru cel mai bun individ
        cities_distances = self.computeIndividDistanceFromCities(tsp_individ)
        # calculeaza timpul pentru cel mai bun individ
        time   = (cities_distances / speeds).sum()
        # calculeaza scorul pentru cel mai bun individ
        score  = profit - R*time
        return score
    # individ metrics =====================

    # population metrics ---------------------
    def computeDistances(self, population):
        """Calculaza distanta pentru fiecare individ din populatiei"""
        return np.apply_along_axis(self.computeIndividDistance,
                                        axis=1,
                                        arr=population)

    def computeNumberCities(self, population):
        """Calculeaza numarul de orase pentru fiecare individ din populatie"""
        return np.apply_along_axis(self.computeIndividNumberCities,
                                        axis=1,
                                        arr=population)

    def computeProfit(self, kp_population):
        """
        Calculeaza profitul pentru intrega populatie
        """
        return np.apply_along_axis(self.computeIndividProfit,
                                        axis=1,
                                        arr=kp_population)

    def computeWeight(self, kp_population):
        """
        Calculeaza profitul pentru intrega populatie
        """
        return np.apply_along_axis(self.computeIndividWeight,
                                        axis=1,
                                        arr=kp_population)

    def computeNbrObj(self, kp_population):
        """
        Calculeaza profitul pentru intrega populatie
        """
        return np.apply_along_axis(self.computeIndividNbrObj,
                                        axis=1,
                                        arr=kp_population)
    # population metrics =====================
