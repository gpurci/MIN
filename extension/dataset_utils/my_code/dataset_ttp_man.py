#!/usr/bin/python

import numpy as np
from extension.dataset_utils.my_code.dataset_base import *

class DatasetTTPMan(DatasetBase):
    """
    """
    def __init__(self, dataset):
        self.dataset = dataset
        # 
        self.__distance    = dataset["distance"]
        self.__item_profit = dataset["item_profit"]
        self.__item_weight = dataset["item_weight"]

    def getTupleDataset(self):
        # unpack datassets
        distance    = self.dataset["distance"]
        item_profit = self.dataset["item_profit"]
        item_weight = self.dataset["item_weight"]
        return (distance, item_profit, item_weight)

    def neighbors(self, size):
        genom_length  = self.dataset["GENOME_LENGTH"]
        x_range       = np.arange(genom_length, dtype=np.int32)
        ret_neighbors = np.argsort(self.__distance[x_range], axis=-1)[:, 1:size+1]
        return ret_neighbors

    # individ  ---------------------
    def computeIndividDistance(self, individ):
        distances = self.__distance[individ[:-1], individ[1:]].sum()
        return distances + self.__distance[individ[-1], individ[0]] # intoarcerea in orasul de start

    def individCityDistance(self, individ):
        city_distances = self.__distance[individ[:-1], individ[1:]]
        to_first_city  = self.__distance[individ[-1], individ[0]]
        return np.concatenate((city_distances, [to_first_city]))

    def computeIndividProfit(self, kp_individ):
        return (self.__item_profit*kp_individ).sum()

    def computeIndividWeight(self, kp_individ):
        return (self.__item_weight*kp_individ).sum()

    def computeIndividNbrObj(self, kp_individ):
        return kp_individ.sum()

    def computeIndividNumberCities(self, individ):
        return np.unique(individ, return_index=False, return_inverse=False, return_counts=False, axis=None).shape[0]
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