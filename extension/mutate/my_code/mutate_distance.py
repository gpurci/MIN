#!/usr/bin/python

import numpy as np
from extension.mutate.my_code.mutate_base import *

class MutateDistance(MutateBase):
    """
    Clasa 'MutateDistance', ofera doar metode pentru a face mutatia genetica a unui individ din populatie.
    Functia 'mutate' are 3 parametri, parinte1, parinte2, descendent.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, dataset_man=None, neighbors_size=20, **configs):
        super().__init__(method, name="MutateDistance", **configs)
        self.__fn = self._unpackMethod(method, 
                                        distance=self.mutateDistance, 
                                    )
        self.dataset_man = dataset_man
        if (dataset_man is not None):
            self.neighbors = dataset_man.neighborsDistance(neighbors_size)
        else:
            self.neighbors = None

    def __call__(self, parent1, parent2, offspring):
        return self.__fn(parent1, parent2, offspring, **self._configs)

    def help(self):
        info = """MutateDistance:
    metoda: 'distance'; config: -> "subset_size":20;
    dataset_man    - managerul setului de date, metode de procesare a cromosomilor
    neighbors_size - numarul celor mai apropiati vecini\n"""
        print(info)

    def mutateDistance(self, parent1, parent2, offspring, subset_size=20):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        # gaseste cei mai apropiati vecini
        offspring_neighbors = all_offspring_neighbors(offspring) # sorted
        offspring_neighbors = np.concatenate((self.neighbors, offspring_neighbors), axis=1)
        # creaza o mapa de vizite
        visited_city      = np.zeros(offspring.shape[0], dtype=bool)
        # calculeaza distanta dintre orasele din chromosom
        city_distances    = self.dataset_man.individCityDistance(offspring)
        # gaseste pozitia celei mai mari distante
        arg_max_distance  = np.argmax(city_distances)
        # seteaza un camp de actiune
        locus1 = max(0,                    arg_max_distance-subset_size)
        locus2 = min(self.GENOME_LENGTH-1, arg_max_distance+subset_size)
        visited_city[:locus1] = True
        visited_city[locus2:] = True
        # minimizare distanta
        tmp = complete(offspring.copy(), visited_city, offspring_neighbors, locus1, locus2, 0, subset_size*5)
        if (tmp is not None):
            offspring = tmp
        # returneaza mostenitorul
        return offspring

    def mutateDistance(self, parent1, parent2, offspring, subset_size=20):
        """Incrucisarea a doi parinti pentru a crea un urmas
        parent1 - individ
        parent2 - individ
        """
        # gaseste cei mai apropiati vecini
        offspring_neighbors = all_offspring_neighbors(offspring) # sorted
        # calculeaza distanta dintre orasele din chromosom
        city_distances    = self.dataset_man.individCityDistance(offspring)
        # gaseste pozitia celei mai mari distante
        arg_max_distance  = np.argmax(city_distances)
        # orasul cel mai indepartat
        far_city  = offspring[arg_max_distance]
        # cei mai apropiati vecini
        near_neighbors_city = self.neighbors[far_city]
        # 
        # returneaza mostenitorul
        return offspring

def find_locus(offspring, near_neighbors_city):
    # cauta pozitia celor mai apropiati vecini in urmasi
    pos_offspring, pos_neighbors = np.nonzero(offspring == near_neighbors_city.reshape(-1, 1))
    # cauta ordinea
    order = pos_neighbors[:-1] - pos_neighbors[1:]
    for cond in [1, -1]:
        mask  = order == cond
        if (mask.sum() > 0):
            arg = np.argmax(mask)
            break
    else:
        arg = near_neighbors_city[0]
    return arg


def complete(offspring, visited_city, offspring_neighbors, locus1, locus2, count, max_count):
    if (locus1 == locus2):
        return offspring
    # selectam orasul de start
    start_city     = offspring[locus1]
    city_neighbors = offspring_neighbors[locus1]
    visited_city[start_city] = True
    # 
    tmp_mask_visited_neighbors = np.invert(visited_city[city_neighbors])
    all_tmp = tmp_mask_visited_neighbors.sum()
    if ((all_tmp == 0) or (count >= max_count)):
        visited_city[start_city] = False
        return None
    else:
        tmp_old_neighbor = offspring[locus1+1]
        for sel_neighbor in city_neighbors[tmp_mask_visited_neighbors]:
            offspring[locus1+1] = int(sel_neighbor)
            tmp = complete(offspring, visited_city, offspring_neighbors, locus1+1, locus2, count+1, max_count)
            if (tmp is not None):
                break
            else:
                offspring[locus1+1] = tmp_old_neighbor
        else:
            offspring[locus1+1] = tmp_old_neighbor
            visited_city[start_city] = False
            return None
    return offspring

def all_offspring_neighbors(offspring):
    # gasseste vecinii
    tmp1 = np.roll(offspring,  1).reshape(1, -1)
    tmp2 = np.roll(offspring, -1).reshape(1, -1)
    neighbors = np.concatenate((tmp1, tmp2), axis=0)
    # sortarea pozitiei oraselor
    sort_pos  = np.argsort(offspring)
    # sortarea vecinilor
    neighbors = neighbors.transpose()[sort_pos]
    return neighbors
