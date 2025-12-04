#!/usr/bin/python

import numpy as np
from extension.crossover.my_code.crossover_base import *
from extension.crossover.my_code.erx_utils import *

class CrossoverDistance(CrossoverBase):
    """
    Clasa 'CrossoverDistance', ofera doar metode pentru a face incrucisarea genetica a doi parinti
    Functia 'crossover' are 2 parametri, parinte1, parinte2.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, dataset_man, neighbors_size=20, **configs):
        super().__init__(method, name="CrossoverDistance", **configs)
        self.__fn = self._unpackMethod(method, 
                                        distance=self.crossover,
                                        distance_all=self.crossoverAll)
        self.dataset_man = dataset_man
        self.neighbors   = dataset_man.neighborsDistance(neighbors_size)

    def __call__(self, parent1, parent2):
        return self.__fn(parent1, parent2, **self._configs)

    def help(self):
        info = """CrossoverDistance:
    metoda: 'distance'; config None;
    metoda: 'distance_all'; config None;
    dataset_man    - managerul setului de date, metode de procesare a cromosomilor
    neighbors_size - numarul celor mai apropiati vecini\n"""
        print(info)

    def crossover(self, parent1, parent2):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        # mosteneste parinte1
        parents_neighbors = neighbors_parents(parent1, parent2) # sorted
        visited_city  = np.zeros(parent1.shape[0], dtype=bool)
        offspring     = parent1.copy()
        GENOME_LENGTH = parent1.shape[0]
        # 
        for idx in range(GENOME_LENGTH-1):
            # selectam orasul de start
            start_city = offspring[idx]
            visited_city[start_city] = True
            # obtinem toti vecinii pentru orasul actual
            city_parent_neighbors = parents_neighbors[start_city]
            city_neighbors        = self.neighbors[start_city]
            # orase potentiale
            tmp_cities, tmp_percent_cities = percent_city(city_parent_neighbors, city_neighbors, visited_city)
            tmp_percent_cities = tmp_percent_cities / tmp_percent_cities.sum()
            # 
            sel_neighbor = np.random.choice(tmp_cities, size=None, p=tmp_percent_cities)
            offspring[idx+1] = int(sel_neighbor)
        # returneaza mostenitorul
        return offspring

    def crossoverAll(self, parent1, parent2):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        # gaseste vecinii
        parents_neighbors = all_neighbors_parents(parent1, parent2) # sorted
        parents_neighbors = np.concatenate((parents_neighbors, self.neighbors), axis=1)
        visited_city  = np.zeros(parent1.shape[0], dtype=bool)
        offspring     = parent1.copy()
        GENOME_LENGTH = parent1.shape[0]
        # 
        for idx in range(GENOME_LENGTH-1):
            # selectam orasul de start
            start_city     = offspring[idx]
            city_neighbors = parents_neighbors[idx]
            visited_city[start_city] = True
            tmp_percent_cities = np.invert(visited_city[city_neighbors]).astype(np.float32)
            all_tmp = tmp_percent_cities.sum()
            if (all_tmp == 0):
                tmp_percent_cities[:] = 1./tmp_percent_cities.shape[0]
            else:
                tmp_percent_cities = tmp_percent_cities / all_tmp
            # 
            sel_neighbor = np.random.choice(city_neighbors, size=None, p=tmp_percent_cities)
            offspring[idx+1] = int(sel_neighbor)
        # returneaza mostenitorul
        return offspring

def percent_city(city_parent_neighbors, city_neighbors, visited_city):
    unique_tmp = np.union1d(city_parent_neighbors, city_neighbors)
    tmp = []
    for u in unique_tmp:
        if (visited_city[u] == False):
            freq = (city_parent_neighbors == u).sum() + (city_neighbors == u).sum()
            tmp.append((u, 1/freq))
    else:
        tmp = np.array(tmp).transpose()
        if (tmp.shape[0] > 0):
            cities, p_cities = tmp[0], tmp[1]
        else:
            cities, p_cities = city_neighbors, np.ones(city_neighbors.shape[0])
    return cities, p_cities

def all_neighbors_parents(parent1, parent2):
    # gasseste vecinii
    tmp1 = np.roll(parent1,  1).reshape(1, -1)
    tmp2 = np.roll(parent1, -1).reshape(1, -1)
    neighbors1 = np.concatenate((tmp1, tmp2), axis=0)
    tmp1 = np.roll(parent2,  1).reshape(1, -1)
    tmp2 = np.roll(parent2, -1).reshape(1, -1)
    neighbors2 = np.concatenate((tmp1, tmp2), axis=0)
    # sortarea pozitiei oraselor
    sort_pos_p1 = np.argsort(parent1)
    sort_pos_p2 = np.argsort(parent2)
    #
    neighbors1 = neighbors1.transpose()[sort_pos_p1]
    neighbors2 = neighbors2.transpose()[sort_pos_p2]
    # 
    neighbors = np.concatenate((neighbors1, neighbors2), axis=1)
    return neighbors