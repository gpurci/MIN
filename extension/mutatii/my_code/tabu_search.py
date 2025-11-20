#!/usr/bin/python

import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import *

class TabuSearch(RootGA):
    """
    Clasa 'TabuSearch', 
    """
    def __init__(self, method, dataset, **configs):
        super().__init__()
        self.__dataset = dataset
        self.__configs = configs
        self.__setMethods(method)

    def __setMethods(self, method):
        self.__method = method
        self.__fn = self.__unpack_method(method)

    def __unpack_method(self, method):
        fn = self.tabuSearchAbstract
        if (method is not None):
            if   (method == "tabu_search"):
                fn = self.tabuSearch
            elif (method == "tabu_search_rand"):
                fn = self.tabuSearchRand
            elif (method == "tabu_search_distance"):
                fn = self.tabuSearchDistance

        return fn

    def __str__(self):
        info = """TabuSearch: 
    method:  {}
    configs: {}
Parent: {}""".format(self.__method, self.__configs, super().__str__())
        return info

    def __call__(self, parent1, parent2, offspring):
        offspring = self.__fn(parent1, parent2, offspring)
        return offspring

    def tabuSearchAbstract(self, parent1, parent2, offspring):
        raise NameError("Lipseste metoda '{}',pentru functia de 'TabuSearch', configs '{}'".format(self.__method, self.__configs))

    def tabuSearch(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        best_score    = self.computeIndividDistance(offspring)
        ret_offspring = offspring.copy()
        for i in range(0, self.GENOME_LENGTH, 1):
            for j in range(i+1, self.GENOME_LENGTH, 1):
                tmp = offspring.copy()
                tmp[i], tmp[j] = tmp[j], tmp[i]
                tmp_score = self.computeIndividDistance(tmp)
                if (best_score > tmp_score):
                    best_score = tmp_score
                    ret_offspring  = tmp
        return ret_offspring

    def tabuSearchRand(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        start = np.random.randint(low=0,                           high=self.GENOME_LENGTH//2, size=None)
        stop  = np.random.randint(low=start+self.GENOME_LENGTH//4, high=self.GENOME_LENGTH,    size=None)

        best_score = self.computeIndividDistance(offspring)
        ret_offspring = offspring.copy()
        for locus1 in range(start, stop, 1):
            for locus2 in range(locus1+1, stop, 1):
                tmp = offspring.copy()
                tmp[locus1], tmp[locus2] = tmp[locus2], tmp[locus1]
                tmp_score = self.computeIndividDistance(tmp)
                if (best_score > tmp_score):
                    best_score = tmp_score
                    ret_offspring = tmp
        return ret_offspring

    def tabuSearchDistance(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        city_distances = self.individCityDistance(offspring)
        d_mean = np.mean(city_distances)
        mask   = city_distances > d_mean
        bad_edges = np.flatnonzero(mask)
        n_bad = bad_edges.shape[0]

        if n_bad < 2:
            return offspring

        # LIMIT neighborhood size
        max_bad = self.__configs.get("max_bad_edges", 40)

        if n_bad > max_bad:
            bad_edges = np.random.choice(bad_edges, size=max_bad, replace=False)
            n_bad = max_bad

        best_score = city_distances.sum()
        ret_offspring = offspring.copy()
        first_improvement = self.__configs.get("first_improvement", True) 
        # explore neighborhood only on selected bad positions
        for idx_i in range(n_bad):
            i = bad_edges[idx_i]
            for idx_j in range(idx_i + 1, n_bad):
                j = bad_edges[idx_j]

                tmp = offspring.copy()
                tmp[i], tmp[j] = tmp[j], tmp[i]

                tmp_score = self.computeIndividDistance(tmp)
                if (best_score > tmp_score):
                    best_score = tmp_score
                    ret_offspring = tmp
                    # major speed-up if we exit early
                    if first_improvement: 
                        return ret_offspring 
        return ret_offspring

    def computeIndividDistance(self, individ):
        """Calculul distantei pentru un individ"""
        #print("individ", individ.shape, end=", ")
        distances = self.__dataset["distance"][individ[:-1], individ[1:]]
        distance  = distances.sum() + self.__dataset["distance"][individ[-1], individ[0]]
        return distance

    def individCityDistance(self, individ):
        """Calculul distantei pentru un individ"""
        #print("individ", individ.shape, end=", ")
        distances = self.__dataset["distance"][individ[:-1], individ[1:]]
        distance  = self.__dataset["distance"][individ[-1], individ[0]]
        return np.concatenate((distances, distance), axis=None)

    def help(self):
        info = """TabuSearch:
    metoda: 'tabu_search';          config: None;
    metoda: 'tabu_search_rand';     config: None;
    metoda: 'tabu_search_distance'; config: None;
    'dataset' - dataset \n"""
        print(info)
