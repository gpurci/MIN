#!/usr/bin/python

import numpy as np
from extension.init_population.init_population_base import *
from extension.utils.insertion import *
from AntColonyOptimizer.ant_colony import *

class InitPopulationHibridACOtspTSkp(InitPopulationBase):
    """
    Clasa 'InitPopulationHibridACOtspTSkp', 
    """
    def __init__(self, method, metrics=None, alpha=1.0, beta=5.0, rho=0.5, q=1.0, **configs):
        super().__init__(method, name="InitPopulationHibridACOtspTSkp", **configs)
        self.__fn = self._unpackMethod(method, 
                                        init=self.init,
                                    )
        self.metrics    = metrics
        self.ant_colony = AntColonyOptimization(metrics.distance, metrics.GENOME_LENGTH, 
                            alpha=alpha, beta=beta, rho=rho, q=q)

    def __call__(self, population_size):
        return self.__fn(population_size, **self._configs)

    def help(self):
        info = """InitPopulationHibridACOtspTSkp:
    metoda: 'init';  config: alpha=1.0, beta=5.0, rho=0.5, q=1.0, city=0, v_min=0.1, v_max=1, W=2000, R=1;
    'metrics' - metrici a setului de date \n"""
        print(info)

    def init(self, population_size=-1, city=0, v_min=0.1, v_max=1, W=2000, R=1):
        """
        """
        if ((population_size == -1) or (population_size >= self.POPULATION_SIZE)):
            population_size = self.POPULATION_SIZE

        tsp_population = self.computeRoute(population_size, city)
        print("Ruta a fost compusa")
        kp_population  = self.computeProfit(population_size, tsp_population, v_min, v_max, W, R)
        print("Profitul a fost compus")
        return {"tsp":np.array(tsp_population, dtype=np.int32), "kp":kp_population}

    def computeRoute(self, population_size, city):
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
        
    # ----------------------------------------------------------------------
    def __call__(self, start_city, generations, size_ants, monitor_size):
        # best-so-far memory
        self.best_path = None
        self.best_cost = np.inf
        self._evolution_monitor = np.zeros(monitor_size, dtype=np.float32)
        for epoch, all_paths in self.ant_colony(start_city, generations, size_ants):
            all_costs = self.metrics.computeDistances(all_paths)
            best_path, best_cost = self._find_best_path(all_paths, all_costs)
            self._update_pheromones(best_path, best_cost)
            print("generatia: {}, best cost: {}".format(epoch, best_cost))
            self._evolution_monitor[:-1] = self._evolution_monitor[1:]
            self._evolution_monitor[-1]  = best_cost
            if (np.allclose(self._evolution_monitor, best_cost, rtol=1e-03, atol=1e-07)):
                break
        return self.best_path
    # ======================================================================
