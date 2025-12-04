#!/usr/bin/python

import numpy as np
from extension.init_population.my_code.init_population_base import *

class RecInitPopulation(InitPopulationBase):
    """
    Clasa 'RecInitPopulation', 
    """
    def __init__(self, method, dataset_man, **configs):
        super().__init__(method, name="RecInitPopulation", **configs)
        self.__fn = self._unpackMethod(method, 
                                        rec_ttp=self.recTTP,
                                    )
        self.__dataset_man = dataset_man

    def __call__(self, population_size):
        return self.__fn(population_size, **self._configs)

    def help(self):
        info = """RecInitPopulation:
    metoda: 'rec_ttp';  config: city=0, window_size=4;
    'dataset_man' - managerul setului de date \n"""
        print(info)

    def recTTP(self, population_size=-1, city=0, window_size=4):
        """
        """
        if ((population_size == -1) or (population_size == self.POPULATION_SIZE)):
            self.__population_size = self.POPULATION_SIZE
            population_size = self.POPULATION_SIZE
        else:
            city        = np.random.randint(low=0, high=self.GENOME_LENGTH, size=None)
            window_size = np.random.randint(low=5, high=20, size=None)
            self.__population_size = population_size

        visited_city = np.zeros(self.GENOME_LENGTH, dtype=bool)
        visited_city[city] = True
        tsp_population   = self.recFill(city, window_size, visited_city, self.GENOME_LENGTH-1)
        for tmp in tsp_population:
            tmp.insert(0, city)
        tsp_population = np.array(tsp_population, dtype=np.int32)
        kp_population  = np.random.randint(low=0, high=2, size=(population_size, self.GENOME_LENGTH))
        return {"tsp":tsp_population, "kp":kp_population}

    def recFill(self, city, window_size, visited_city, deep):
        if (deep == 0):
            self.__population_size -= 1
            return [[]]
        if (self.__population_size <= 0):
            return None
        args = self.__dataset_man.unvisitedNeighborDistance(city, window_size, visited_city)
        #print("visited_city", np.argwhere(visited_city).reshape(-1), "actual city", city)
        #print("args", args, "deep", deep)
        population = []
        for arg in args:
            visited_city[arg] = True
            tmp_individs = self.recFill(arg, window_size, visited_city, deep-1)
            visited_city[arg] = False
            #print(tmp_individs)
            if (tmp_individs is not None):
                for tmp in tmp_individs:
                    tmp.insert(0, arg)
                population.extend(tmp_individs)
        visited_city[args] = False
        return population

