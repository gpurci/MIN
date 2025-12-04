#!/usr/bin/python

import numpy as np
from extension.mutate.mutate_base import *

class MutateInsertion(MutateBase):
    """
    Clasa 'Mutate', ofera doar metode pentru a face mutatia genetica a unui individ din populatie.
    Functia 'mutate' are 3 parametri, parinte1, parinte2, descendent.
    Metoda 'call', returneaza functia din configuratie.
    Pentru o configuratie inexistenta, vei primi un mesaj de eroare.
    """
    def __init__(self, method, dataset_man=None, neighbors_size=20, **configs):
        super().__init__(method, name="MutateInsertion", **configs)
        self.__fn = self._unpackMethod(method, 
                                        insertion=self.insertion, 
                                        distance=self.distance
                                        )
        self.dataset_man = dataset_man
        if (dataset_man is not None):
            self.neighbors = dataset_man.neighborsDistance(neighbors_size)
        else:
            self.neighbors = None

    def __call__(self, parent1, parent2, offspring):
        return self.__fn(parent1, parent2, offspring, **self._configs)

    def help(self):
        info = """MutateInsertion:
    metoda: 'insertion'; config: None;\n"""
        print(info)

    def insertion(self, parent1, parent2, offspring):
        """Mutatia genetica a indivizilor, operatie in_place
            parent1 - individul parinte 1
            parent2 - individul parinte 2
            offspring - individul copil/descendent
        """
        locus1 = np.random.randint(low=0,      high=self.GENOME_LENGTH//2, size=None)
        locus2 = np.random.randint(low=locus1, high=self.GENOME_LENGTH,    size=None)
        # copy gene
        offspring = negativeInsertion(offspring, locus1, locus2)
        return offspring

    def distance(self, parent1, parent2, offspring):
        """Incrucisarea a doi parinti pentru a crea un urmas
            parent1 - individ
            parent2 - individ
            offspring - individul copil/descendent
        """
        # gaseste cei mai apropiati vecini
        offspring_neighbors = all_offspring_neighbors(offspring) # sorted
        # calculeaza distanta dintre orasele din chromosom
        city_distances      = self.dataset_man.individCityDistance(offspring)
        # gaseste pozitia celei mai mari distante
        locus1   = np.argmax(city_distances)
        # orasul cel mai indepartat
        far_city = offspring[locus1]
        # cei mai apropiati vecini
        near_neighbors_city = self.neighbors[far_city]
        # 
        locus2 = find_locus(offspring, near_neighbors_city)
        #
        if (locus1 > locus2):
            offspring = positiveInsertion(offspring, locus1, locus2)
        else:
            offspring = negativeInsertion(offspring, locus1, locus2)
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

def positiveInsertion(offspring, locus1, locus2):
    gene    = offspring[locus1]
    # make change locuses
    locuses = np.arange(locus2+1, locus1)
    offspring[locuses+1] = offspring[locuses]
    offspring[locus2+1]  = gene
    return offspring

def negativeInsertion(offspring, locus1, locus2):
    gene    = offspring[locus1]
    # make change locuses
    locuses = np.arange(locus1, locus2)
    offspring[locuses] = offspring[locuses+1]
    offspring[locus2]  = gene
    return offspring
