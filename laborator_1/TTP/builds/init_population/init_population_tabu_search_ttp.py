#!/usr/bin/python

import numpy as np
from extension.init_population.init_population_base import *
from extension.utils.insertion import *

class InitPopulationTabuSearch(InitPopulationBase):
    """
    Clasa 'InitPopulationTabuSearch', 
    """
    def __init__(self, method, dataset_man=None, **configs):
        super().__init__(method, name="InitPopulationTabuSearch", **configs)
        self.__fn = self._unpackMethod(method, 
                                        init=self.init,
                                    )
        self.dataset_man = dataset_man

    def __call__(self, population_size):
        return self.__fn(population_size, **self._configs)

    def help(self):
        info = """InitPopulationTabuSearch:
    metoda: 'init';  config: city=0, window_size=4, v_min=0.1, v_max=1, W=2000, R=1;
    'dataset_man' - managerul setului de date \n"""
        print(info)

    def init(self, population_size=-1, city=0, window_size=4, v_min=0.1, v_max=1, W=2000, R=1):
        """
        """
        if ((population_size == -1) or (population_size == self.POPULATION_SIZE)):
            self.__population_size = self.POPULATION_SIZE
            population_size        = self.POPULATION_SIZE

        tsp_population = self.computeRoute(population_size, city, window_size)
        kp_population  = self.computeProfit(population_size, tsp_population, v_min, v_max, W, R)
        return {"tsp":np.array(tsp_population, dtype=np.int32), "kp":kp_population}

    def computeRoute(self, population_size, city, window_size):
        self.__population_size = population_size
        # creaza mapa de orase vizitate
        visited_city = np.zeros(self.GENOME_LENGTH, dtype=bool)
        visited_city[city] = True
        # creaza rute baza cel mai apropiat vecin
        tsp_population = self.recNeighborFill(city, window_size, visited_city, self.GENOME_LENGTH-1)
        # adauga orasul de start
        for route in tsp_population:
            route.insert(0, city)
        tsp_population = np.array(tsp_population, dtype=np.int32)
        #print("primul oras", tsp_population[:, 0], " city", city)
        # aplica tabu search pe rute
        for idx in range(population_size):
            route = tsp_population[idx]
            is_find = True
            while (is_find): # cauta cea mai buna ruta,
                route, is_find = self.insertion_tabu_search_distance(route, city)
            tsp_population[idx] = route
        return np.array(tsp_population, dtype=np.int32)

    def recNeighborFill(self, city, window_size, visited_city, deep):
        if (deep == 0):
            self.__population_size -= 1
            return [[]]
        if (self.__population_size <= 0):
            return None
        tmp_window_size = np.random.randint(low=2, high=window_size, size=None)
        args = self.dataset_man.unvisitedNeighborDistance(city, tmp_window_size, visited_city)
        #print("visited_city", np.argwhere(visited_city).reshape(-1), "actual city", city)
        #print("args", args, "deep", deep)
        population = []
        for arg in args:
            visited_city[arg] = True
            tmp_individs = self.recNeighborFill(arg, window_size, visited_city, deep-1)
            visited_city[arg] = False
            #print(tmp_individs)
            if (tmp_individs is not None):
                for tmp in tmp_individs:
                    tmp.insert(0, arg)
                population.extend(tmp_individs)
        visited_city[args] = False
        return population

    def insertion_tabu_search_distance(self, tsp_individ, city):
        # calcularea distantelor dintre fiecare oras
        city_distances = self.dataset_man.individCityDistance(tsp_individ)
        #city_distances[city] = 0
        # creare mask de depasire media pe distanta
        locus1 = np.argmax(city_distances) + 1
        if (locus1 >= self.GENOME_LENGTH):
            locus1 = self.GENOME_LENGTH-1
        # compute score
        best_distance = city_distances.sum()
        best_individ  = tsp_individ.copy()
        # apply tabu search
        locusses = np.arange(self.GENOME_LENGTH, dtype=np.int32)
        locusses = np.delete(locusses, obj=[city, locus1])
        #print("city {}, locusses {}".format(city, locusses))
        # set flag 
        is_find = False
        # find best distance
        for locus2 in locusses:
            #print("locus1 {}, locus2 {}".format(locus1, locus2))
            tmp = insertion(tsp_individ.copy(), locus1, locus2)
            distance = self.dataset_man.computeIndividDistance(tmp)
            if (distance < best_distance):
                best_distance = distance
                best_individ  = tmp
                is_find = True
        # set best route
        return best_individ, is_find

    def computeProfit(self, population_size, tsp_population, v_min, v_max, W, R):
        kp_population = np.random.randint(low=0, high=2, size=(population_size, self.GENOME_LENGTH))
        # sterge ccele mai grele obiecte
        for idx in range(population_size):
            kp_population[idx] = self.erase_weightier_objects(kp_population[idx], W)
        # calculeaza cea mai buna combinatie, de obiecte
        # aplica tabu search pe profit
        is_find = True
        for idx in range(population_size):
            route  = tsp_population[idx]
            profit = kp_population[idx]
            last_score = 0
            while (is_find): # cauta cea mai buna ruta,
                profit, is_find = self.tabu_search_score(route, profit, v_min, v_max, W, R)
                tmp_profit = obj.erase_weightier_objects(profit, W)
                tmp_score  = dataset_obj.computeIndividScore(route, tmp_profit, v_min=v_min, v_max=v_max, W=W, R=R)
                if (tmp_score < last_score):
                    break
                else:
                    profit = tmp_profit
                last_score = tmp_score

            kp_population[idx] = profit
        return np.array(kp_population, dtype=np.int32)

    def erase_weightier_objects(self, kp_individ, W):
        while (self.dataset_man.computeIndividWeight(kp_individ) > W):
            argmax = self.dataset_man.argIndividMaxWeight(kp_individ)
            kp_individ[argmax] = 0
        return kp_individ

    def tabu_search_score(self, tsp_individ, kp_individ, v_min, v_max, W, R):
        # calcularea distantelor dintre fiecare oras
        mask = kp_individ==1
        arg_take   = np.argwhere(mask).reshape(-1)
        arg_notake = np.argwhere(np.invert(mask)).reshape(-1)
        # compute score
        best_score   = self.dataset_man.computeIndividScore(tsp_individ, kp_individ, v_min=v_min, v_max=v_max, W=W, R=R)
        best_individ = kp_individ.copy()
        # apply tabu search
        is_find = False
        for locus1 in arg_take:
            for locus2 in arg_notake:
                tmp = kp_individ.copy()
                tmp[locus1] = 0
                tmp[locus2] = 1
                score  = self.dataset_man.computeIndividScore(tsp_individ, tmp, v_min=v_min, v_max=v_max, W=W, R=R)
                if ((score > best_score)):
                    best_score   = score
                    best_individ = tmp.copy()
                    is_find = True
        # set best route
        return best_individ, is_find

