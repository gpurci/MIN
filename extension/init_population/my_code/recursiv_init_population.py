#!/usr/bin/python

import numpy as np
from extension.init_population.my_code.init_population_base import *

class RecInitPopulation(InitPopulationBase):
    """
    Clasa 'RecInitPopulation', 
    """
    def __init__(self, method, dataset, **configs):
        super().__init__(method, name="RecInitPopulation", **configs)
        self.__fn = self._unpackMethod(method, 
                                        rec_ttp=self.recTTP,
                                    )
        self.__dataset = dataset

    def __call__(self, size, genoms=None):
        self.__fn(size, genoms=genoms, **self._configs)

    def help(self):
        info = """RecInitPopulation:
    metoda: 'rec_ttp';  config: "city":0, "window_size":4;
    'dataset' - dataset \n"""
        print(info)

    def recTTP(self, population_size=-1, genoms=None, city=0, window_size=4):
        """
        """
        if ((population_size == -1) or (population_size == self.POPULATION_SIZE)):
            self.__population_size = self.POPULATION_SIZE
        else:
            city        = np.random.randint(low=0, high=self.GENOME_LENGTH, size=None)
            window_size = np.random.randint(low=5, high=20, size=None)
            self.__population_size = population_size

        visited_city = np.zeros(self.GENOME_LENGTH, dtype=bool)
        visited_city[city] = True
        population   = self.recFill(city, window_size, visited_city, self.GENOME_LENGTH-1)
        for tmp in population:
            tmp.insert(0, city)
        population = np.array(population, dtype=np.int32)
        
        if (genoms is not None):
            for tsp_individ in population:
                # adauga tsp_individ in genome
                kp_individ = np.random.randint(low=0, high=2, size=self.GENOME_LENGTH)
                genoms.add(tsp=tsp_individ, kp=kp_individ)
            # adauga indivizi in noua generatie
            genoms.saveInit()
            print("recTTP", genoms)
        else:
            raise NameError("Din functia externa 'RecInitPopulation', metoda 'recTTP', lipseste 'genoms'")

    def computeBestKDistance(self, city, window_size, visited_city):
        """Calculul distantei pentru un individ"""
        #print("individ", individ.shape, end=", ")
        distances = self.__dataset["distance"][city]
        # 
        args  = np.argsort(distances)
        count = 0
        ret_args = []
        for pos_city in args:
            if (visited_city[pos_city] == False):
                ret_args.append(pos_city)
                count += 1
            if (count >= window_size):
                break
        return np.array(ret_args, dtype=np.int32)

    def recFill(self, city, window_size, visited_city, deep):
        if (deep == 0):
            self.__population_size -= 1
            return [[]]
        if (self.__population_size <= 0):
            return None
        args = self.computeBestKDistance(city, window_size, visited_city)
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

