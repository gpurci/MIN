#!/usr/bin/python

import numpy as np
from GeneticAlgorithmManager.my_code.root_GA import *

class TwoOpt(RootGA):
    """
    Clasa 'TwoOpt', 
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
        fn = self.twoOptAbstract

        if method is not None:
            if   (method == "two_opt"):
                fn = self.twoOpt
            elif (method == "two_opt_rand"):
                fn = self.twoOptRand
            elif (method == "two_opt_distance"):
                fn = self.twoOptDistance
            elif (method == "two_opt_LS"):
                fn = self.twoOptLS

        return fn


    def __str__(self):
        info = """TwoOpt: 
    method:  {}
    configs: {}
Parent: {}""".format(self.__method, self.__configs, super().__str__())
        return info
    
    def __call__(self, *args):
        """
        SAFE universal signature:

        Accepted calls:
        - op(route)                      → LS mode
        - op(p1, p2, route)             → GA mode
        - op(None, None, route)         → elite LS via TTP-VND
        """

        # ------- CASE 1: Local-search TwoOptLS(route) -------
        if self.__method == "two_opt_LS":
            if len(args) == 1:
                # correct LS-style call: op(route)
                (route,) = args
                return self.twoOptLS(route)
            elif len(args) == 3:
                # GA-style call but LS method → treat as LS
                _, _, route = args
                return self.twoOptLS(route)
            else:
                raise ValueError(f"TwoOptLS expected 1 or 3 args, got {len(args)}")

        # ------- CASE 2: All other methods use GA interface -------
        if len(args) != 3:
            raise ValueError(f"{self.__method} expects THREE arguments (p1,p2,offspring). Got {len(args)}.")

        parent1, parent2, offspring = args
        return self.__fn(parent1, parent2, offspring)

    def twoOptAbstract(self, parent1, parent2, offspring):
        raise NameError("Lipseste metoda '{}',pentru functia de 'TwoOpt', configs '{}'".format(self.__method, self.__configs))

    def twoOpt(self, parent1, parent2, offspring):
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

    def twoOptRand(self, parent1, parent2, offspring):
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

    def twoOptDistance(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        city_distances = self.individCityDistance(offspring)
        d_mean = np.mean(city_distances)
        mask   = city_distances > d_mean
        args_distances = np.argwhere(mask)

        best_score = city_distances.sum()
        ret_offspring = offspring.copy()
        for i in range(0, args_distances.shape[0], 1):
            for j in range(i+1, args_distances.shape[0], 1):
                tmp = offspring.copy()
                locus1 = args_distances[i]
                locus2 = args_distances[j]
                tmp[locus1], tmp[locus2] = tmp[locus2], tmp[locus1]
                tmp_score = self.computeIndividDistance(tmp)
                if (best_score > tmp_score):
                    best_score = tmp_score
                    ret_offspring = tmp
        return ret_offspring
    
    def twoOptLS(self, route):
        """
        Local-search TwoOpt for elite improvement.
        Does NOT require parent1/parent2/offspring.
        Signature:
            improved_route = twoOptLS(route)
        """
        best = route.copy()
        best_d = self.computeIndividDistance(best)
        n = len(best)

        # optional iteration count from configs
        iters = self.__configs.get("iters", 50)

        for _ in range(iters):
            i = np.random.randint(1, n - 2)
            k = np.random.randint(i + 1, n - 1)

            new_r = best.copy()
            new_r[i:k+1] = new_r[i:k+1][::-1]

            d = self.computeIndividDistance(new_r)
            if d < best_d:
                best, best_d = new_r, d

        return best

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
        info = """TwoOpt:
    metoda: 'two_opt';          config: None;
    metoda: 'two_opt_rand';     config: None;
    metoda: 'two_opt_distance'; config: None;
    'dataset' - dataset \n"""
        print(info)
